"""TensorFlow / Keras model parser.

Extracts layer configurations from ``tf.keras.Model.layers`` and converts
them into ``List[LayerInfo]`` without requiring a forward pass.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

from roofline.core.layer_info import LayerInfo
from roofline.parsers.base import BaseParser

# Keras class name → normalised layer_type
_TF_TYPE_MAP: Dict[str, str] = {
    "Dense": "Linear",
    "Conv1D": "Conv1d",
    "Conv2D": "Conv2d",
    "Conv3D": "Conv3d",
    "Conv1DTranspose": "ConvTranspose1d",
    "Conv2DTranspose": "ConvTranspose2d",
    "Conv3DTranspose": "ConvTranspose3d",
    "DepthwiseConv2D": "Conv2d",
    "SeparableConv2D": "Conv2d",
    "MultiHeadAttention": "MultiHeadAttention",
    "BatchNormalization": "BatchNorm",
    "LayerNormalization": "LayerNorm",
    "GroupNormalization": "GroupNorm",
    "InstanceNormalization": "InstanceNorm",
    "Embedding": "Embedding",
    "LSTM": "LSTM",
    "GRU": "GRU",
    "SimpleRNN": "RNN",
    "LSTMCell": "LSTM",
    "GRUCell": "GRU",
    "Bidirectional": "LSTM",
    "Activation": "Activation",
    "ReLU": "ReLU",
    "ELU": "ELU",
    "LeakyReLU": "ReLU",
    "Softmax": "Softmax",
    "MaxPooling1D": "MaxPool",
    "MaxPooling2D": "MaxPool",
    "MaxPooling3D": "MaxPool",
    "AveragePooling1D": "AvgPool",
    "AveragePooling2D": "AvgPool",
    "AveragePooling3D": "AvgPool",
    "GlobalAveragePooling1D": "AdaptiveAvgPool",
    "GlobalAveragePooling2D": "AdaptiveAvgPool",
    "GlobalMaxPooling2D": "AdaptiveMaxPool",
    "Flatten": "Reshape",
    "Reshape": "Reshape",
    "Dropout": "Dropout",
    "Add": "Elementwise",
    "Multiply": "Elementwise",
    "Concatenate": "Concat",
}


class TensorFlowParser(BaseParser):
    """Parse a ``tf.keras.Model`` into ``List[LayerInfo]``."""

    def parse(
        self,
        model,
        input_shapes: Optional[List[Tuple]] = None,
        dtype: str = "float32",
    ) -> List[LayerInfo]:
        try:
            import tensorflow as tf
        except ImportError as e:
            raise ImportError(
                "tensorflow is required for TF/Keras parsing. "
                "Install with: pip install tensorflow"
            ) from e

        if isinstance(model, str):
            model = tf.keras.models.load_model(model)

        if not isinstance(model, tf.keras.Model):
            raise TypeError(f"TensorFlowParser expects a tf.keras.Model, got {type(model)}")

        # Build the model with dummy input if needed (to populate output shapes)
        if input_shapes:
            try:
                import numpy as np
                dummy = [np.zeros(sh, dtype=np.float32) for sh in input_shapes]
                model(dummy[0] if len(dummy) == 1 else dummy, training=False)
            except Exception:
                pass

        layers: List[LayerInfo] = []
        for layer in model.layers:
            layers.extend(self._process_layer(layer, dtype))

        return layers

    def _process_layer(self, layer, dtype: str) -> List[LayerInfo]:
        """Recursively process nested layers (e.g., Sequential, Functional)."""
        try:
            import tensorflow as tf
            # Recurse into nested models
            if isinstance(layer, tf.keras.Model) and layer.layers:
                result = []
                for sub in layer.layers:
                    result.extend(self._process_layer(sub, dtype))
                return result
        except Exception:
            pass

        cls_name = type(layer).__name__
        layer_type = _TF_TYPE_MAP.get(cls_name, cls_name)

        # Count parameters
        num_params = sum(
            int(_prod(w.shape)) for w in layer.weights if "kernel" in w.name or "weight" in w.name
        )

        # Input / output shapes
        in_shapes: List[Tuple] = []
        out_shapes: List[Tuple] = []
        try:
            if hasattr(layer, "input_shape"):
                raw = layer.input_shape
                if isinstance(raw, (list, tuple)) and raw:
                    if isinstance(raw[0], (list, tuple)):
                        in_shapes = [tuple(s) for s in raw]
                    else:
                        in_shapes = [tuple(raw)]
        except Exception:
            pass
        try:
            if hasattr(layer, "output_shape"):
                raw = layer.output_shape
                if isinstance(raw, (list, tuple)) and raw:
                    if isinstance(raw[0], (list, tuple)):
                        out_shapes = [tuple(s) for s in raw]
                    else:
                        out_shapes = [tuple(raw)]
        except Exception:
            pass

        cfg = _safe_get_config(layer)
        attrs = _extract_keras_attrs(cls_name, cfg, in_shapes, out_shapes)

        return [
            LayerInfo(
                name=layer.name,
                layer_type=layer_type,
                input_shapes=in_shapes,
                output_shapes=out_shapes,
                num_params=num_params,
                dtype=dtype,
                attrs=attrs,
                source_framework="tensorflow",
            )
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prod(shape) -> int:
    result = 1
    for d in shape:
        if d is not None:
            result *= int(d)
    return result


def _safe_get_config(layer) -> Dict[str, Any]:
    try:
        return layer.get_config()
    except Exception:
        return {}


def _extract_keras_attrs(cls_name: str, cfg: Dict, in_shapes, out_shapes) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}

    if cls_name == "Dense":
        attrs["in_features"] = in_shapes[0][-1] if in_shapes else cfg.get("units")
        attrs["out_features"] = cfg.get("units")
        attrs["bias"] = cfg.get("use_bias", True)

    elif cls_name in ("Conv1D", "Conv2D", "Conv3D", "DepthwiseConv2D", "SeparableConv2D"):
        attrs["out_channels"] = cfg.get("filters")
        ks = cfg.get("kernel_size", 3)
        attrs["kernel_size"] = ks if isinstance(ks, (list, tuple)) else (ks, ks)
        attrs["strides"] = cfg.get("strides", (1, 1))
        attrs["padding"] = cfg.get("padding", "valid")
        attrs["groups"] = cfg.get("groups", 1)
        attrs["bias"] = cfg.get("use_bias", True)
        if in_shapes:
            attrs["in_channels"] = in_shapes[0][-1]  # channels-last

    elif cls_name == "MultiHeadAttention":
        attrs["num_heads"] = cfg.get("num_heads")
        attrs["embed_dim"] = cfg.get("key_dim", cfg.get("value_dim"))

    elif cls_name in ("LSTM", "GRU", "SimpleRNN"):
        attrs["hidden_size"] = cfg.get("units")
        attrs["return_sequences"] = cfg.get("return_sequences", False)
        if in_shapes:
            attrs["input_size"] = in_shapes[0][-1]

    elif cls_name == "Embedding":
        attrs["num_embeddings"] = cfg.get("input_dim")
        attrs["embedding_dim"] = cfg.get("output_dim")

    elif cls_name == "LayerNormalization":
        attrs["axis"] = cfg.get("axis")

    elif cls_name == "BatchNormalization":
        attrs["axis"] = cfg.get("axis")

    return attrs
