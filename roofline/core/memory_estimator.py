"""MemoryEstimator — computes memory traffic per layer.

Two modes:

* ``"inference"`` — weight bytes + activation bytes (input + output tensors)
* ``"training"``  — above + gradient bytes (≈ activations) + optimizer state bytes
                    (Adam: 2 × params × dtype_bytes)
"""

from __future__ import annotations

import warnings
from typing import Tuple

from roofline.core.layer_info import LayerInfo, dtype_bytes


class MemoryEstimator:
    """Estimates memory traffic (bytes) for a single layer."""

    # Optimizer state multiplier for Adam (momentum + variance) stored in fp32
    _ADAM_STATE_BYTES_PER_PARAM = 8  # 2 × float32

    def estimate(
        self,
        layer: LayerInfo,
        mode: str = "inference",
        dtype: str = None,
    ) -> Tuple[int, int, int, int]:
        """Return ``(weight_bytes, activation_bytes, grad_bytes, optimizer_bytes)``.

        Parameters
        ----------
        layer:
            The ``LayerInfo`` to estimate for.
        mode:
            ``"inference"`` or ``"training"``.
        dtype:
            Override the dtype for byte-width calculation.
            Defaults to ``layer.dtype``.
        """
        dtype = dtype or layer.dtype
        bpe = dtype_bytes(dtype)   # bytes per element

        weight_bytes = self._weight_bytes(layer, bpe)
        activation_bytes = self._activation_bytes(layer, bpe)

        grad_bytes = 0
        optimizer_bytes = 0

        if mode == "training":
            # Gradients: same shape as weights, stored in same dtype
            grad_bytes = weight_bytes
            # Optimizer states: Adam stores 2 fp32 tensors per parameter
            optimizer_bytes = layer.num_params * self._ADAM_STATE_BYTES_PER_PARAM

        return weight_bytes, activation_bytes, grad_bytes, optimizer_bytes

    # ------------------------------------------------------------------
    # Weight bytes
    # ------------------------------------------------------------------

    def _weight_bytes(self, layer: LayerInfo, bpe: float) -> int:
        """Bytes to load all layer parameters from memory."""
        if layer.num_params > 0:
            return int(layer.num_params * bpe)

        # Fallback: estimate from attrs
        attrs = layer.attrs
        lt = layer.layer_type

        if lt == "Linear":
            in_f = attrs.get("in_features") or 0
            out_f = attrs.get("out_features") or 0
            params = in_f * out_f
            if attrs.get("bias"):
                params += out_f
            return int(params * bpe)

        if lt in ("Conv1d", "Conv2d", "Conv3d", "ConvTranspose2d"):
            Cin = attrs.get("in_channels") or 0
            Cout = attrs.get("out_channels") or 0
            ks = attrs.get("kernel_size", (1,))
            if isinstance(ks, int):
                ks = (ks,)
            groups = attrs.get("groups", 1) or 1
            kernel_size = 1
            for k in ks:
                if k:
                    kernel_size *= k
            params = (Cin // groups) * Cout * kernel_size
            if attrs.get("bias"):
                params += Cout
            return int(params * bpe)

        if lt in ("BatchNorm", "LayerNorm", "GroupNorm"):
            num_features = (
                attrs.get("num_features")
                or (attrs.get("normalized_shape") or [0])[0] if isinstance(
                    attrs.get("normalized_shape"), (list, tuple)) else attrs.get("num_features", 0)
            )
            return int(num_features * 2 * bpe)  # weight + bias

        if lt == "Embedding":
            ne = attrs.get("num_embeddings") or 0
            de = attrs.get("embedding_dim") or 0
            return int(ne * de * bpe)

        if lt == "LSTM":
            H = attrs.get("hidden_size") or 0
            I = attrs.get("input_size") or 0
            layers_n = attrs.get("num_layers", 1) or 1
            dirs = 2 if attrs.get("bidirectional") else 1
            # 4 gates × (I+H) × H × directions × layers
            params = 4 * (I + H) * H * dirs * layers_n
            if attrs.get("bias"):
                params += 4 * H * dirs * layers_n * 2
            return int(params * bpe)

        if lt == "GRU":
            H = attrs.get("hidden_size") or 0
            I = attrs.get("input_size") or 0
            layers_n = attrs.get("num_layers", 1) or 1
            dirs = 2 if attrs.get("bidirectional") else 1
            params = 3 * (I + H) * H * dirs * layers_n
            return int(params * bpe)

        return 0

    # ------------------------------------------------------------------
    # Activation bytes
    # ------------------------------------------------------------------

    def _activation_bytes(self, layer: LayerInfo, bpe: float) -> int:
        """Bytes for input + output activation tensors."""
        in_elems = sum(_prod(s) for s in layer.input_shapes)
        out_elems = sum(_prod(s) for s in layer.output_shapes)
        return int((in_elems + out_elems) * bpe)


def _prod(shape) -> int:
    result = 1
    for d in shape:
        if d is not None and d > 0:
            result *= int(d)
    return result
