"""PyTorch model parser.

Uses ``torch.fx.symbolic_trace`` to walk the computation graph and extract
per-layer ``LayerInfo`` records.  Falls back to a hook-based approach for
models that are not fx-traceable (e.g., models with data-dependent control
flow).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

from roofline.core.layer_info import LayerInfo
from roofline.parsers.base import BaseParser

# Mapping from PyTorch module class names → normalised layer_type strings
_TYPE_MAP: Dict[str, str] = {
    "Linear": "Linear",
    "Bilinear": "Linear",
    "LazyLinear": "Linear",
    "Conv1d": "Conv1d",
    "Conv2d": "Conv2d",
    "Conv3d": "Conv3d",
    "ConvTranspose1d": "ConvTranspose1d",
    "ConvTranspose2d": "ConvTranspose2d",
    "ConvTranspose3d": "ConvTranspose3d",
    "MultiheadAttention": "MultiHeadAttention",
    "BatchNorm1d": "BatchNorm",
    "BatchNorm2d": "BatchNorm",
    "BatchNorm3d": "BatchNorm",
    "SyncBatchNorm": "BatchNorm",
    "LayerNorm": "LayerNorm",
    "GroupNorm": "GroupNorm",
    "InstanceNorm1d": "InstanceNorm",
    "InstanceNorm2d": "InstanceNorm",
    "InstanceNorm3d": "InstanceNorm",
    "Embedding": "Embedding",
    "EmbeddingBag": "Embedding",
    "LSTM": "LSTM",
    "GRU": "GRU",
    "RNN": "RNN",
    "ReLU": "ReLU",
    "ReLU6": "ReLU",
    "GELU": "GELU",
    "SiLU": "SiLU",
    "Sigmoid": "Sigmoid",
    "Tanh": "Tanh",
    "Softmax": "Softmax",
    "LogSoftmax": "Softmax",
    "Dropout": "Dropout",
    "MaxPool1d": "MaxPool",
    "MaxPool2d": "MaxPool",
    "MaxPool3d": "MaxPool",
    "AvgPool1d": "AvgPool",
    "AvgPool2d": "AvgPool",
    "AvgPool3d": "AvgPool",
    "AdaptiveAvgPool1d": "AdaptiveAvgPool",
    "AdaptiveAvgPool2d": "AdaptiveAvgPool",
    "AdaptiveAvgPool3d": "AdaptiveAvgPool",
    "Flatten": "Reshape",
    "Unflatten": "Reshape",
}


class PyTorchParser(BaseParser):
    """Parse a ``torch.nn.Module`` into ``List[LayerInfo]``.

    Tries ``torch.fx.symbolic_trace`` first; if that fails (e.g., dynamic
    control flow), falls back to a forward-hook pass that runs one sample
    inference to capture shapes.
    """

    def parse(
        self,
        model,
        input_shapes: Optional[List[Tuple]] = None,
        dtype: str = "float32",
    ) -> List[LayerInfo]:
        import torch

        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"PyTorchParser expects a torch.nn.Module, got {type(model)}")

        model.eval()

        # Try fx tracing first
        layers = self._parse_via_fx(model, input_shapes, dtype)
        if layers:
            return layers

        # Fallback: hook-based
        if input_shapes is None:
            warnings.warn(
                "Could not fx-trace model and no input_shapes provided. "
                "Layer shapes will be unknown.",
                stacklevel=2,
            )
            return self._parse_via_named_modules(model, dtype)

        return self._parse_via_hooks(model, input_shapes, dtype)

    # ------------------------------------------------------------------
    # fx trace approach
    # ------------------------------------------------------------------

    def _parse_via_fx(self, model, input_shapes, dtype) -> List[LayerInfo]:
        try:
            import torch
            from torch import fx

            concrete_args = {}
            traced = fx.symbolic_trace(model)
            return self._extract_from_traced(traced, model, input_shapes, dtype)
        except Exception:
            return []

    def _extract_from_traced(self, traced, original_model, input_shapes, dtype):
        import torch
        import io, contextlib
        layers: List[LayerInfo] = []
        name_to_module = dict(original_model.named_modules())

        # 1. Run ShapeProp to populate node.meta for nodes that accept float32.
        #    Suppress noise from ops that need non-float inputs (e.g. Embedding).
        if input_shapes:
            try:
                from torch.fx.passes.shape_prop import ShapeProp
                sample_inputs_f32 = [
                    torch.zeros(*sh, dtype=torch.float32) for sh in input_shapes
                ]
                sp = ShapeProp(traced)
                with contextlib.redirect_stderr(io.StringIO()):
                    sp.propagate(*sample_inputs_f32)
            except Exception:
                pass

        # 2. Run hook-based forward pass to capture real shapes for ALL leaf modules.
        #    This covers ops like Embedding that ShapeProp can't handle with float32 inputs.
        hook_shapes: Dict[str, tuple] = {}
        if input_shapes:
            hook_shapes = self._capture_shapes_via_hooks(original_model, input_shapes)

        for node in traced.graph.nodes:
            if node.op != "call_module":
                continue
            mod = name_to_module.get(node.target)
            if mod is None:
                continue

            layer_type = _TYPE_MAP.get(type(mod).__name__, type(mod).__name__)
            num_params = sum(p.numel() for p in mod.parameters())

            # Get shapes: prefer hook-captured, fall back to ShapeProp meta
            if node.target in hook_shapes:
                in_shapes, out_shapes = hook_shapes[node.target]
            else:
                out_shapes = _shapes_from_node_meta(node)
                in_shapes = []
                for arg in node.args:
                    if hasattr(arg, "meta"):
                        in_shapes.extend(_shapes_from_node_meta(arg))

            attrs = _extract_attrs(mod)

            layers.append(
                LayerInfo(
                    name=node.target,
                    layer_type=layer_type,
                    input_shapes=in_shapes,
                    output_shapes=out_shapes,
                    num_params=num_params,
                    dtype=dtype,
                    attrs=attrs,
                    source_framework="pytorch",
                )
            )

        return layers

    def _capture_shapes_via_hooks(
        self, model, input_shapes
    ) -> Dict[str, tuple]:
        """Run a forward pass with hooks to capture (in_shapes, out_shapes) per module."""
        import torch
        shapes: Dict[str, tuple] = {}
        hooks = []

        def _make_hook(name):
            def hook(module, inp, out):
                in_s = [tuple(t.shape) for t in inp if isinstance(t, torch.Tensor)]
                if isinstance(out, torch.Tensor):
                    out_s = [tuple(out.shape)]
                elif isinstance(out, (tuple, list)):
                    out_s = [tuple(t.shape) for t in out if isinstance(t, torch.Tensor)]
                else:
                    out_s = []
                shapes[name] = (in_s, out_s)
            return hook

        for name, mod in model.named_modules():
            if len(list(mod.children())) == 0:  # leaf only
                hooks.append(mod.register_forward_hook(_make_hook(name)))

        try:
            # Use appropriate dtypes: long for first input if model has Embedding
            has_embedding = any(
                isinstance(m, torch.nn.Embedding)
                for m in model.modules()
            )
            if has_embedding:
                sample = [torch.zeros(*sh, dtype=torch.long) for sh in input_shapes]
            else:
                sample = [torch.zeros(*sh, dtype=torch.float32) for sh in input_shapes]
            with torch.no_grad():
                model(*sample)
        except Exception:
            pass
        finally:
            for h in hooks:
                h.remove()

        return shapes

    # ------------------------------------------------------------------
    # Hook-based approach (fallback)
    # ------------------------------------------------------------------

    def _parse_via_hooks(self, model, input_shapes, dtype) -> List[LayerInfo]:
        import torch

        layers: List[LayerInfo] = []
        hooks = []

        dtype_torch = _dtype_str_to_torch(dtype)

        def make_hook(name, mod):
            def hook(module, inp, out):
                in_shapes = [tuple(t.shape) for t in inp if isinstance(t, torch.Tensor)]
                if isinstance(out, torch.Tensor):
                    out_shapes = [tuple(out.shape)]
                elif isinstance(out, (tuple, list)):
                    out_shapes = [tuple(t.shape) for t in out if isinstance(t, torch.Tensor)]
                else:
                    out_shapes = []

                layer_type = _TYPE_MAP.get(type(module).__name__, type(module).__name__)
                num_params = sum(p.numel() for p in module.parameters())
                attrs = _extract_attrs(module)

                layers.append(
                    LayerInfo(
                        name=name,
                        layer_type=layer_type,
                        input_shapes=in_shapes,
                        output_shapes=out_shapes,
                        num_params=num_params,
                        dtype=dtype,
                        attrs=attrs,
                        source_framework="pytorch",
                    )
                )
            return hook

        # Register hooks on leaf modules only
        for name, mod in model.named_modules():
            if len(list(mod.children())) == 0:  # leaf
                h = mod.register_forward_hook(make_hook(name, mod))
                hooks.append(h)

        try:
            sample_inputs = [
                torch.zeros(*sh, dtype=dtype_torch) for sh in input_shapes
            ]
            with torch.no_grad():
                model(*sample_inputs)
        except Exception as e:
            warnings.warn(f"Forward pass failed during hook-based parsing: {e}", stacklevel=2)
        finally:
            for h in hooks:
                h.remove()

        return layers

    # ------------------------------------------------------------------
    # Named-modules fallback (no shapes)
    # ------------------------------------------------------------------

    def _parse_via_named_modules(self, model, dtype) -> List[LayerInfo]:
        layers = []
        for name, mod in model.named_modules():
            if len(list(mod.children())) > 0:
                continue  # skip container modules
            layer_type = _TYPE_MAP.get(type(mod).__name__, type(mod).__name__)
            num_params = sum(p.numel() for p in mod.parameters())
            attrs = _extract_attrs(mod)
            layers.append(
                LayerInfo(
                    name=name or type(mod).__name__,
                    layer_type=layer_type,
                    input_shapes=[],
                    output_shapes=[],
                    num_params=num_params,
                    dtype=dtype,
                    attrs=attrs,
                    source_framework="pytorch",
                )
            )
        return layers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shapes_from_node_meta(node) -> List[Tuple]:
    """Extract output shape(s) from an fx node's meta dict (populated by ShapeProp)."""
    try:
        meta = node.meta.get("tensor_meta") or node.meta.get("val")
        if meta is not None:
            if hasattr(meta, "shape"):
                return [tuple(int(d) for d in meta.shape)]
            # Some versions return a namedtuple with .shape
            if hasattr(meta, "dtype") and hasattr(meta, "shape"):
                return [tuple(int(d) for d in meta.shape)]
    except Exception:
        pass
    return []


def _shapes_from_node(node, input_shapes):
    """Best-effort shape extraction from an fx node (legacy fallback)."""
    return [], _shapes_from_node_meta(node)


def _extract_attrs(mod) -> Dict[str, Any]:
    """Extract relevant attributes from a module for FlopCounter."""
    attrs: Dict[str, Any] = {}
    cls = type(mod).__name__

    if cls in ("Linear", "LazyLinear", "Bilinear"):
        attrs["in_features"] = getattr(mod, "in_features", None)
        attrs["out_features"] = getattr(mod, "out_features", None)
        attrs["bias"] = mod.bias is not None

    elif cls.startswith("Conv"):
        attrs["in_channels"] = getattr(mod, "in_channels", None)
        attrs["out_channels"] = getattr(mod, "out_channels", None)
        ks = getattr(mod, "kernel_size", None)
        attrs["kernel_size"] = ks if isinstance(ks, (tuple, list)) else (ks, ks)
        attrs["groups"] = getattr(mod, "groups", 1)
        attrs["stride"] = getattr(mod, "stride", (1,))
        attrs["padding"] = getattr(mod, "padding", (0,))
        attrs["bias"] = mod.bias is not None

    elif cls == "MultiheadAttention":
        attrs["embed_dim"] = getattr(mod, "embed_dim", None)
        attrs["num_heads"] = getattr(mod, "num_heads", None)
        attrs["dropout"] = getattr(mod, "dropout", 0.0)
        attrs["bias"] = getattr(mod, "in_proj_bias", None) is not None
        attrs["batch_first"] = getattr(mod, "batch_first", False)

    elif cls in ("BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "SyncBatchNorm"):
        attrs["num_features"] = getattr(mod, "num_features", None)
        attrs["affine"] = getattr(mod, "affine", True)

    elif cls == "LayerNorm":
        attrs["normalized_shape"] = getattr(mod, "normalized_shape", None)
        attrs["elementwise_affine"] = getattr(mod, "elementwise_affine", True)

    elif cls in ("Embedding", "EmbeddingBag"):
        attrs["num_embeddings"] = getattr(mod, "num_embeddings", None)
        attrs["embedding_dim"] = getattr(mod, "embedding_dim", None)

    elif cls in ("LSTM", "GRU", "RNN"):
        attrs["input_size"] = getattr(mod, "input_size", None)
        attrs["hidden_size"] = getattr(mod, "hidden_size", None)
        attrs["num_layers"] = getattr(mod, "num_layers", 1)
        attrs["bidirectional"] = getattr(mod, "bidirectional", False)
        attrs["bias"] = getattr(mod, "bias", True)

    return attrs


def _dtype_str_to_torch(dtype: str):
    import torch
    _map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "float64": torch.float64,
    }
    return _map.get(dtype.lower(), torch.float32)
