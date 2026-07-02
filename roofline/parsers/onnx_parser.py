"""ONNX model parser.

Walks the ONNX ``NodeProto`` graph to extract per-operator ``LayerInfo``.
Shape inference is run via ``onnx.shape_inference.infer_shapes`` so that
input/output tensor shapes are available for memory estimation.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

from roofline.core.layer_info import LayerInfo
from roofline.parsers.base import BaseParser

# ONNX op_type → normalised layer_type
_OP_MAP: Dict[str, str] = {
    "Gemm": "Linear",
    "MatMul": "MatMul",
    "Conv": "Conv2d",
    "ConvTranspose": "ConvTranspose2d",
    "BatchNormalization": "BatchNorm",
    "InstanceNormalization": "InstanceNorm",
    "LayerNormalization": "LayerNorm",
    "GroupNormalization": "GroupNorm",
    "Relu": "ReLU",
    "Gelu": "GELU",
    "Sigmoid": "Sigmoid",
    "Tanh": "Tanh",
    "Softmax": "Softmax",
    "LogSoftmax": "Softmax",
    "Gather": "Embedding",
    "GatherElements": "Embedding",
    "LSTM": "LSTM",
    "GRU": "GRU",
    "RNN": "RNN",
    "MaxPool": "MaxPool",
    "AveragePool": "AvgPool",
    "GlobalAveragePool": "AdaptiveAvgPool",
    "GlobalMaxPool": "AdaptiveMaxPool",
    "Flatten": "Reshape",
    "Reshape": "Reshape",
    "Squeeze": "Reshape",
    "Unsqueeze": "Reshape",
    "Transpose": "Transpose",
    "Add": "Elementwise",
    "Mul": "Elementwise",
    "Sub": "Elementwise",
    "Div": "Elementwise",
    "Dropout": "Dropout",
    "Attention": "MultiHeadAttention",
    "MultiHeadAttention": "MultiHeadAttention",
}


class ONNXParser(BaseParser):
    """Parse an ONNX model file or ``onnx.ModelProto`` into ``List[LayerInfo]``."""

    def parse(
        self,
        model,
        input_shapes: Optional[List[Tuple]] = None,
        dtype: str = "float32",
    ) -> List[LayerInfo]:
        try:
            import onnx
        except ImportError as e:
            raise ImportError(
                "onnx is required for ONNX parsing. Install with: pip install onnx"
            ) from e

        if isinstance(model, str):
            proto = onnx.load(model)
        elif hasattr(model, "graph"):
            proto = model
        else:
            raise TypeError(f"ONNXParser expects a path string or ModelProto, got {type(model)}")

        # Run shape inference
        try:
            proto = onnx.shape_inference.infer_shapes(proto)
        except Exception:
            warnings.warn("ONNX shape inference failed; shapes may be unknown.", stacklevel=2)

        # Build value_info lookup: name → shape
        shape_info: Dict[str, Tuple] = {}
        for vi in list(proto.graph.input) + list(proto.graph.value_info) + list(proto.graph.output):
            shape = _extract_onnx_shape(vi)
            if shape:
                shape_info[vi.name] = shape

        # Override first input shape if provided
        if input_shapes and proto.graph.input:
            shape_info[proto.graph.input[0].name] = input_shapes[0]

        # Build initializer (weight) name set for param counting
        initializer_shapes: Dict[str, Tuple] = {}
        for init in proto.graph.initializer:
            initializer_shapes[init.name] = tuple(init.dims)

        layers: List[LayerInfo] = []
        node_counts: Dict[str, int] = {}

        for node in proto.graph.node:
            op = node.op_type
            layer_type = _OP_MAP.get(op, op)

            # Unique name
            base = node.name or f"{op}_{len(layers)}"
            node_counts[base] = node_counts.get(base, 0) + 1
            name = base if node_counts[base] == 1 else f"{base}_{node_counts[base]}"

            in_shapes = [shape_info[i] for i in node.input if i in shape_info]
            out_shapes = [shape_info[o] for o in node.output if o in shape_info]

            # Count params from weight initializers referenced by this node
            num_params = 0
            for inp_name in node.input:
                if inp_name in initializer_shapes:
                    p = 1
                    for d in initializer_shapes[inp_name]:
                        p *= d
                    num_params += p

            attrs = _parse_onnx_attrs(node, op, in_shapes, out_shapes)

            layers.append(
                LayerInfo(
                    name=name,
                    layer_type=layer_type,
                    input_shapes=in_shapes,
                    output_shapes=out_shapes,
                    num_params=num_params,
                    dtype=dtype,
                    attrs=attrs,
                    source_framework="onnx",
                )
            )

        return layers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_onnx_shape(value_info) -> Optional[Tuple]:
    try:
        import onnx
        t = value_info.type.tensor_type
        if not t.HasField("shape"):
            return None
        dims = []
        for d in t.shape.dim:
            if d.HasField("dim_value"):
                dims.append(d.dim_value)
            else:
                dims.append(None)
        return tuple(dims) if dims else None
    except Exception:
        return None


def _parse_onnx_attrs(node, op: str, in_shapes, out_shapes) -> Dict[str, Any]:
    """Extract relevant numeric attributes from an ONNX node."""
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        return {}

    attrs: Dict[str, Any] = {}

    def get_attr(name):
        for a in node.attribute:
            if a.name == name:
                if a.type == onnx.AttributeProto.INT:
                    return a.i
                elif a.type == onnx.AttributeProto.FLOAT:
                    return a.f
                elif a.type == onnx.AttributeProto.INTS:
                    return list(a.ints)
                elif a.type == onnx.AttributeProto.FLOATS:
                    return list(a.floats)
                elif a.type == onnx.AttributeProto.STRING:
                    return a.s.decode()
        return None

    if op == "Gemm":
        attrs["transA"] = get_attr("transA") or 0
        attrs["transB"] = get_attr("transB") or 0
        if in_shapes and len(in_shapes) >= 1:
            s = in_shapes[0]
            attrs["in_features"] = s[-1] if s else None
        if out_shapes:
            s = out_shapes[0]
            attrs["out_features"] = s[-1] if s else None

    elif op == "Conv":
        attrs["kernel_shape"] = get_attr("kernel_shape") or []
        attrs["strides"] = get_attr("strides") or [1, 1]
        attrs["pads"] = get_attr("pads") or [0, 0, 0, 0]
        attrs["group"] = get_attr("group") or 1
        attrs["dilations"] = get_attr("dilations") or [1, 1]
        if in_shapes and len(in_shapes[0]) >= 2:
            attrs["in_channels"] = in_shapes[0][1]
        if out_shapes and len(out_shapes[0]) >= 2:
            attrs["out_channels"] = out_shapes[0][1]

    elif op in ("LSTM", "GRU", "RNN"):
        attrs["hidden_size"] = get_attr("hidden_size")
        attrs["direction"] = get_attr("direction") or "forward"
        if in_shapes:
            attrs["input_size"] = in_shapes[0][-1] if in_shapes[0] else None

    elif op in ("Attention", "MultiHeadAttention"):
        attrs["num_heads"] = get_attr("num_heads")

    elif op == "LayerNormalization":
        attrs["axis"] = get_attr("axis")
        attrs["epsilon"] = get_attr("epsilon")

    return attrs
