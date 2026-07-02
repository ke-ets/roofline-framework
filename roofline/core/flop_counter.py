"""FlopCounter — registry-based FLOPs estimation per layer type.

Each handler receives a ``LayerInfo`` and returns an integer FLOPs count.
New layer types can be registered without modifying core logic:

    from roofline.core.flop_counter import FlopCounter
    fc = FlopCounter()

    @fc.register("MyCustomOp")
    def count_my_op(layer):
        return layer.attrs["n"] * 2
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Dict

from roofline.core.layer_info import LayerInfo

Handler = Callable[[LayerInfo], int]


class FlopCounter:
    """Counts FLOPs for each layer type using a pluggable handler registry."""

    def __init__(self):
        self._registry: Dict[str, Handler] = {}
        self._register_defaults()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count(self, layer: LayerInfo) -> int:
        """Return FLOPs for a single layer.

        Falls back to 0 with a warning if no handler is registered.
        """
        handler = self._registry.get(layer.layer_type)
        if handler is None:
            # Try case-insensitive lookup
            for key, fn in self._registry.items():
                if key.lower() == layer.layer_type.lower():
                    handler = fn
                    break

        if handler is None:
            return 0  # No-op / unknown ops contribute 0 FLOPs

        try:
            return max(0, int(handler(layer)))
        except Exception as e:
            warnings.warn(
                f"FlopCounter handler for '{layer.layer_type}' raised: {e}. "
                f"Returning 0 for layer '{layer.name}'.",
                stacklevel=2,
            )
            return 0

    def count_all(self, layers) -> Dict[str, int]:
        """Return {layer.name: flops} dict for an iterable of LayerInfo."""
        return {layer.name: self.count(layer) for layer in layers}

    def register(self, layer_type: str):
        """Decorator to register a custom handler."""
        def decorator(fn: Handler) -> Handler:
            self._registry[layer_type] = fn
            return fn
        return decorator

    # ------------------------------------------------------------------
    # Default handlers
    # ------------------------------------------------------------------

    def _register_defaults(self):
        reg = self._registry

        # ---- Linear / Dense / Gemm / MatMul ----
        def _linear(layer: LayerInfo) -> int:
            a = layer.attrs
            in_f = a.get("in_features")
            out_f = a.get("out_features")
            if in_f and out_f:
                batch = _batch(layer)
                flops = 2 * batch * in_f * out_f
                if a.get("bias"):
                    flops += batch * out_f
                return flops
            # Fallback: infer from shapes
            return _matmul_from_shapes(layer)

        reg["Linear"] = _linear
        reg["MatMul"] = _matmul_from_shapes

        def _gemm(layer: LayerInfo) -> int:
            # Shape is (M, K) × (K, N) — second input is usually transposed
            if len(layer.input_shapes) >= 2:
                a_shape = layer.input_shapes[0]
                b_shape = layer.input_shapes[1]
                M = _prod(a_shape[:-1])
                K = a_shape[-1] or 1
                N = b_shape[-1] if not layer.attrs.get("transB") else (b_shape[0] if b_shape else 1)
                N = N or 1
                return 2 * M * K * N
            return _linear(layer)

        reg["Gemm"] = _gemm

        # ---- Convolutions ----
        def _conv(layer: LayerInfo, dims: int = 2) -> int:
            a = layer.attrs
            Cin = a.get("in_channels")
            Cout = a.get("out_channels")
            groups = a.get("groups", 1) or 1
            ks = a.get("kernel_size", (1,) * dims)
            if isinstance(ks, int):
                ks = (ks,) * dims

            # Spatial output size
            spatial_out = _spatial_output(layer, dims)

            if Cin and Cout and all(k > 0 for k in ks):
                kernel_ops = int(_prod(ks))
                flops = 2 * (Cin // groups) * Cout * kernel_ops * spatial_out
                if a.get("bias"):
                    flops += Cout * spatial_out
                return flops

            # Fallback from shapes
            if layer.output_shapes:
                out_s = layer.output_shapes[0]
                if len(out_s) >= dims + 2:
                    b = out_s[0] or 1
                    c_out = out_s[1] or 1
                    sp = _prod(out_s[2:])
                    c_in = layer.input_shapes[0][1] if layer.input_shapes else 1
                    kernel_ops = int(_prod(ks)) if ks else 1
                    return 2 * b * c_in * c_out * kernel_ops * sp
            return 0

        reg["Conv1d"] = lambda l: _conv(l, 1)
        reg["Conv2d"] = lambda l: _conv(l, 2)
        reg["Conv3d"] = lambda l: _conv(l, 3)

        def _conv_transpose(layer: LayerInfo, dims: int = 2) -> int:
            # Same as conv but spatial output is (input - 1) * stride + kernel
            return _conv(layer, dims)

        reg["ConvTranspose1d"] = lambda l: _conv_transpose(l, 1)
        reg["ConvTranspose2d"] = lambda l: _conv_transpose(l, 2)
        reg["ConvTranspose3d"] = lambda l: _conv_transpose(l, 3)

        # ---- Multi-Head Attention ----
        def _mha(layer: LayerInfo) -> int:
            a = layer.attrs
            B = _batch(layer)
            embed_dim = a.get("embed_dim") or a.get("d_model")
            num_heads = a.get("num_heads")

            # Infer from input shapes if attrs missing
            if not embed_dim and layer.input_shapes:
                embed_dim = layer.input_shapes[0][-1]
            if not num_heads:
                num_heads = 8  # conservative default

            if not embed_dim:
                return 0

            # Sequence length
            seq_len = 1
            if layer.input_shapes:
                s = layer.input_shapes[0]
                seq_len = s[1] if len(s) >= 2 else 1
            seq_len = seq_len or 1

            head_dim = embed_dim // num_heads

            # Q, K, V projections: 3 × 2 × B × S × D²
            qkv_flops = 3 * 2 * B * seq_len * embed_dim * embed_dim

            # Scaled dot-product attention per head: 2 × B × H × S² × head_dim
            attn_flops = 2 * B * num_heads * (seq_len ** 2) * head_dim

            # Softmax: B × H × S (negligible but included)
            softmax_flops = B * num_heads * seq_len * seq_len * 5  # exp+sum+div

            # Output projection: 2 × B × S × D²
            out_proj_flops = 2 * B * seq_len * embed_dim * embed_dim

            return qkv_flops + attn_flops + softmax_flops + out_proj_flops

        reg["MultiHeadAttention"] = _mha

        # ---- Normalisation ----
        def _norm(layer: LayerInfo) -> int:
            """BatchNorm / LayerNorm / GroupNorm — 2 ops per element (mean+var pass)."""
            elems = _total_elements(layer)
            return 2 * elems if elems else 0

        reg["BatchNorm"] = _norm
        reg["LayerNorm"] = _norm
        reg["GroupNorm"] = _norm
        reg["InstanceNorm"] = _norm

        # ---- Embedding ----
        # Lookups: 0 FLOPs (memory-only)
        reg["Embedding"] = lambda l: 0

        # ---- Element-wise activations: 1 op per element ----
        def _activation(layer: LayerInfo) -> int:
            return _total_elements(layer)

        reg["ReLU"] = _activation
        reg["GELU"] = lambda l: 8 * _total_elements(l)   # GELU ~8 ops (tanh approx)
        reg["SiLU"] = lambda l: 4 * _total_elements(l)
        reg["Sigmoid"] = lambda l: 4 * _total_elements(l)
        reg["Tanh"] = lambda l: 6 * _total_elements(l)
        reg["Softmax"] = lambda l: 5 * _total_elements(l)
        reg["Activation"] = _activation
        reg["ELU"] = lambda l: 4 * _total_elements(l)

        # ---- Pooling: 1 op per element per kernel position ----
        def _pool(layer: LayerInfo) -> int:
            elems = _total_elements_output(layer)
            ks = layer.attrs.get("kernel_size", (2, 2))
            if isinstance(ks, int):
                ks = (ks, ks)
            kernel_ops = int(_prod(ks)) if ks else 4
            return elems * kernel_ops

        reg["MaxPool"] = _pool
        reg["AvgPool"] = _pool
        reg["AdaptiveAvgPool"] = lambda l: _total_elements_output(l)
        reg["AdaptiveMaxPool"] = lambda l: _total_elements_output(l)

        # ---- Recurrent ----
        def _lstm(layer: LayerInfo) -> int:
            a = layer.attrs
            B = _batch(layer)
            H = a.get("hidden_size") or 0
            I = a.get("input_size") or 0
            T = _seq_len(layer)
            layers_n = a.get("num_layers", 1) or 1
            directions = 2 if a.get("bidirectional") else 1

            if not H or not I:
                return 0

            # 4 gates × (I×H + H×H + H)  × 2 (FMA) per timestep
            per_step = 4 * (2 * I * H + 2 * H * H + H)
            return B * T * per_step * layers_n * directions

        def _gru(layer: LayerInfo) -> int:
            a = layer.attrs
            B = _batch(layer)
            H = a.get("hidden_size") or 0
            I = a.get("input_size") or 0
            T = _seq_len(layer)
            layers_n = a.get("num_layers", 1) or 1
            directions = 2 if a.get("bidirectional") else 1

            if not H or not I:
                return 0

            # 3 gates
            per_step = 3 * (2 * I * H + 2 * H * H + H)
            return B * T * per_step * layers_n * directions

        reg["LSTM"] = _lstm
        reg["GRU"] = _gru
        reg["RNN"] = lambda l: _gru(l)  # single gate RNN ~ GRU without reset

        # ---- Elementwise ops ----
        reg["Elementwise"] = lambda l: _total_elements(l)
        reg["Concat"] = lambda l: _total_elements_output(l)
        reg["Transpose"] = lambda l: 0       # memory movement, not compute
        reg["Reshape"] = lambda l: 0
        reg["Dropout"] = lambda l: 0

        # ---- Attention variants used in modern LLMs ----
        reg["Attention"] = _mha  # same formula
        reg["FlashAttention"] = _mha


# ---------------------------------------------------------------------------
# Shape utilities used by handlers
# ---------------------------------------------------------------------------

def _prod(shape) -> int:
    result = 1
    for d in shape:
        if d is not None and d > 0:
            result *= int(d)
    return result


def _batch(layer: LayerInfo) -> int:
    if layer.input_shapes:
        b = layer.input_shapes[0][0]
        return int(b) if b and b > 0 else 1
    if layer.output_shapes:
        b = layer.output_shapes[0][0]
        return int(b) if b and b > 0 else 1
    return 1


def _seq_len(layer: LayerInfo) -> int:
    if layer.input_shapes and len(layer.input_shapes[0]) >= 2:
        s = layer.input_shapes[0][1]
        return int(s) if s and s > 0 else 1
    return 1


def _total_elements(layer: LayerInfo) -> int:
    if layer.input_shapes:
        return _prod(layer.input_shapes[0])
    if layer.output_shapes:
        return _prod(layer.output_shapes[0])
    return 0


def _total_elements_output(layer: LayerInfo) -> int:
    if layer.output_shapes:
        return _prod(layer.output_shapes[0])
    return _total_elements(layer)


def _spatial_output(layer: LayerInfo, dims: int) -> int:
    if layer.output_shapes:
        s = layer.output_shapes[0]
        if len(s) >= dims + 2:
            b = s[0] or 1
            sp = _prod(s[2:]) or 1
            return int(b) * int(sp)
    # Fallback: assume square spatial output
    if layer.input_shapes:
        s = layer.input_shapes[0]
        if len(s) >= dims + 2:
            return _prod(s[2:]) or 1
    return 1


def _matmul_from_shapes(layer: LayerInfo) -> int:
    """Estimate FLOPs for a matrix multiply from input tensor shapes."""
    shapes = layer.input_shapes
    if len(shapes) >= 2:
        a, b = shapes[0], shapes[1]
        if a and b:
            M = _prod(a[:-1]) or 1
            K = a[-1] or 1
            N = b[-1] or 1
            return 2 * M * K * N
    elif len(shapes) == 1:
        s = shapes[0]
        if len(s) >= 2:
            return 2 * _prod(s[:-1]) * s[-1]
    return 0
