"""Framework-agnostic layer descriptors and per-layer analysis results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class LayerInfo:
    """Framework-agnostic description of a single model layer.

    All parsers (PyTorch, ONNX, HuggingFace, TF/Keras, Folder) produce
    a ``List[LayerInfo]`` as their output.
    """

    name: str
    """Qualified layer name, e.g. ``"layer1.0.conv1"``."""

    layer_type: str
    """Normalised op type string.
    Examples: ``"Linear"``, ``"Conv2d"``, ``"MultiHeadAttention"``,
    ``"LayerNorm"``, ``"Gemm"`` (ONNX), ``"MatMul"`` (ONNX).
    """

    input_shapes: List[Tuple[int, ...]]
    """Shapes of all input tensors, e.g. ``[(1, 512)]``."""

    output_shapes: List[Tuple[int, ...]]
    """Shapes of all output tensors."""

    num_params: int
    """Total number of trainable parameters in this layer."""

    dtype: str
    """Precision string: ``"float32"``, ``"float16"``, ``"bfloat16"``,
    ``"int8"``, or ``"int4"``."""

    attrs: Dict[str, Any] = field(default_factory=dict)
    """Layer-specific attributes used by FlopCounter / MemoryEstimator.

    Examples:
        - Linear:  ``{"in_features": 4096, "out_features": 4096, "bias": True}``
        - Conv2d:  ``{"in_channels": 64, "out_channels": 128,
                      "kernel_size": (3, 3), "groups": 1,
                      "padding": (1,1), "stride": (1,1)}``
        - MHA:     ``{"num_heads": 12, "embed_dim": 768, "dropout": 0.1}``
        - LSTM:    ``{"input_size": 256, "hidden_size": 512, "num_layers": 1,
                      "bidirectional": False}``
    """

    source_framework: str = "unknown"
    """Which parser produced this record: ``"pytorch"``, ``"onnx"``,
    ``"huggingface"``, ``"tensorflow"``."""


# ---------------------------------------------------------------------------
# Per-dtype bytes per element
# ---------------------------------------------------------------------------

DTYPE_BYTES: Dict[str, float] = {
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "int4": 0.5,
}


def dtype_bytes(dtype: str) -> float:
    """Return bytes per element for a given dtype string."""
    key = dtype.lower().replace(" ", "")
    if key not in DTYPE_BYTES:
        raise ValueError(
            f"Unknown dtype '{dtype}'. Supported: {list(DTYPE_BYTES.keys())}"
        )
    return DTYPE_BYTES[key]


# ---------------------------------------------------------------------------
# LayerStats — output of the roofline analysis for one layer
# ---------------------------------------------------------------------------

@dataclass
class LayerStats:
    """Roofline analysis result for a single layer against a specific hardware target."""

    layer: LayerInfo
    """The underlying layer descriptor."""

    # --- compute & memory ---------------------------------------------------
    flops: int
    """Total floating-point operations (MACs × 2, or element-wise ops)."""

    weight_bytes: int
    """Bytes required to load all layer parameters from memory."""

    activation_bytes: int
    """Bytes for input + output activation tensors."""

    grad_bytes: int = 0
    """Gradient bytes (training mode only; 0 for inference)."""

    optimizer_bytes: int = 0
    """Optimizer state bytes (training mode only; 0 for inference)."""

    # --- derived memory totals ----------------------------------------------
    @property
    def total_bytes(self) -> int:
        """Total memory traffic = weights + activations + grads + optimizer."""
        return self.weight_bytes + self.activation_bytes + self.grad_bytes + self.optimizer_bytes

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs per byte of memory traffic (X-axis of roofline plot)."""
        if self.total_bytes == 0:
            return 0.0
        return self.flops / self.total_bytes

    # --- roofline results (populated by RooflineAnalyzer) -------------------
    ridge_point: float = 0.0
    """peak_flops / peak_mem_bw for the target hardware + dtype."""

    attainable_perf: float = 0.0
    """min(AI × peak_mem_bw, peak_flops) — Y-axis value on roofline plot.
    Units: FLOPs/second."""

    bottleneck: str = "unknown"
    """``"compute"`` if AI > ridge_point, else ``"memory"``."""

    theoretical_time_ms: float = 0.0
    """Estimated minimum execution time: flops / attainable_perf × 1000."""

    hw_name: str = ""
    """Name of the hardware target used for this analysis."""

    dtype_used: str = ""
    """Dtype used during analysis (determines which peak_flops entry is used)."""

    def efficiency_pct(self) -> float:
        """Percentage of theoretical peak compute utilised (0–100)."""
        peak = self.attainable_perf
        if peak == 0:
            return 0.0
        actual_perf = self.flops / max(self.theoretical_time_ms / 1000.0, 1e-12)
        return min(actual_perf / peak * 100.0, 100.0)

    def summary_dict(self) -> Dict[str, Any]:
        """Return a flat dict for tabular display / DataFrame construction."""
        return {
            "name": self.layer.name,
            "type": self.layer.layer_type,
            "dtype": self.layer.dtype,
            "params": self.layer.num_params,
            "flops": self.flops,
            "weight_bytes": self.weight_bytes,
            "activation_bytes": self.activation_bytes,
            "grad_bytes": self.grad_bytes,
            "optimizer_bytes": self.optimizer_bytes,
            "total_bytes": self.total_bytes,
            "arithmetic_intensity": round(self.arithmetic_intensity, 4),
            "ridge_point": round(self.ridge_point, 2),
            "attainable_perf_tflops": round(self.attainable_perf / 1e12, 4),
            "efficiency_pct": round(self.efficiency_pct(), 2),
            "bottleneck": self.bottleneck,
            "theoretical_time_ms": round(self.theoretical_time_ms, 6),
            "hw": self.hw_name,
        }
