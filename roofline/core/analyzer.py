"""RooflineAnalyzer — orchestrates parsing, FLOPs counting, memory
estimation, and roofline computation into an ``AnalysisResults`` object.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple

from roofline.core.flop_counter import FlopCounter
from roofline.core.layer_info import LayerInfo, LayerStats
from roofline.core.memory_estimator import MemoryEstimator
from roofline.hardware.hw_spec import HWSpec


class AnalysisResults:
    """Container for the per-layer roofline analysis output.

    Attributes
    ----------
    layers : List[LayerStats]
        One entry per model layer, in execution order.
    hw : HWSpec
        The hardware target used for analysis.
    dtype : str
        The precision used.
    mode : str
        ``"inference"`` or ``"training"``.
    """

    def __init__(
        self,
        layers: List[LayerStats],
        hw: HWSpec,
        dtype: str,
        mode: str,
    ):
        self.layers = layers
        self.hw = hw
        self.dtype = dtype
        self.mode = mode

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def total_flops(self) -> int:
        return sum(l.flops for l in self.layers)

    @property
    def total_params(self) -> int:
        return sum(l.layer.num_params for l in self.layers)

    @property
    def total_weight_bytes(self) -> int:
        return sum(l.weight_bytes for l in self.layers)

    @property
    def total_activation_bytes(self) -> int:
        return sum(l.activation_bytes for l in self.layers)

    @property
    def total_bytes(self) -> int:
        return sum(l.total_bytes for l in self.layers)

    @property
    def theoretical_time_ms(self) -> float:
        return sum(l.theoretical_time_ms for l in self.layers)

    def memory_bound_layers(self) -> List[LayerStats]:
        return [l for l in self.layers if l.bottleneck == "memory"]

    def compute_bound_layers(self) -> List[LayerStats]:
        return [l for l in self.layers if l.bottleneck == "compute"]

    # ------------------------------------------------------------------
    # Reporting shortcuts
    # ------------------------------------------------------------------

    def print_table(self, top_n: Optional[int] = None, sort_by: str = "flops") -> None:
        """Print a rich table of per-layer results."""
        from roofline.reporting.table_report import TableReport
        TableReport(self).print_rich_table(top_n=top_n, sort_by=sort_by)

    def to_dataframe(self):
        """Return a pandas DataFrame of per-layer results."""
        from roofline.reporting.table_report import TableReport
        return TableReport(self).to_dataframe()

    def plot_roofline(
        self,
        figsize=(12, 7),
        save_path: Optional[str] = None,
        show: bool = True,
        annotate_top_n: int = 5,
        model_name: Optional[str] = None,
    ):
        """Draw the roofline chart."""
        from roofline.reporting.roofline_plot import RooflinePlot
        return RooflinePlot(self).plot(
            figsize=figsize,
            save_path=save_path,
            show=show,
            annotate_top_n=annotate_top_n,
            model_name=model_name,
        )

    def print_summary(self) -> None:
        """Print a brief text summary to stdout."""
        print(self.summary())

    def print_table(self) -> None:
        """Print the per-layer table to stdout (rich if available, plain otherwise)."""
        from roofline.reporting.table_report import TableReport
        TableReport(self).print_rich_table()

    def summary(self) -> str:
        """Return a brief text summary."""
        nb = len(self.memory_bound_layers())
        cb = len(self.compute_bound_layers())
        total = len(self.layers)
        ridge = self.hw.ridge_point(self.dtype)
        lines = [
            f"Model analysis on {self.hw.name} [{self.dtype}, {self.mode}]",
            f"  Layers analyzed  : {total}",
            f"  Total FLOPs      : {_fmt_flops(self.total_flops)}",
            f"  Total params     : {_fmt_params(self.total_params)}",
            f"  Total memory     : {_fmt_bytes(self.total_bytes)}",
            f"  Est. total time  : {self.theoretical_time_ms:.3f} ms",
            f"  Ridge point      : {ridge:.1f} FLOPs/Byte",
            f"  Memory-bound     : {nb}/{total} layers ({100*nb//total if total else 0}%)",
            f"  Compute-bound    : {cb}/{total} layers ({100*cb//total if total else 0}%)",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return f"AnalysisResults({len(self.layers)} layers, hw={self.hw.name}, dtype={self.dtype})"


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class RooflineAnalyzer:
    """Orchestrates the full analysis pipeline."""

    def __init__(self):
        self.flop_counter = FlopCounter()
        self.memory_estimator = MemoryEstimator()

    def analyze(
        self,
        model,
        input_shapes: Optional[List[Tuple]] = None,
        hw: Optional[HWSpec] = None,
        dtype: str = "float32",
        mode: str = "inference",
        source: str = None,
        batch_size: int = None,
        fetch_from_web: bool = False,
        detect_local_hw: bool = False,
    ) -> AnalysisResults:
        # ---- 1. Resolve hardware ----
        if hw is None:
            from roofline.hardware.hw_resolver import HWResolver
            hw = HWResolver(fetch_from_web=fetch_from_web).resolve(detect_local=detect_local_hw)

        # ---- 2. Apply batch_size override ----
        if batch_size is not None and input_shapes:
            input_shapes = [(batch_size, *s[1:]) for s in input_shapes]

        # ---- 3. Parse model into LayerInfo list ----
        layer_infos = self._parse(model, source, input_shapes, dtype)

        # ---- 4. Compute FLOPs + memory per layer ----
        layer_stats = []
        ridge = hw.ridge_point(dtype)

        for li in layer_infos:
            flops = self.flop_counter.count(li)
            w_bytes, a_bytes, g_bytes, o_bytes = self.memory_estimator.estimate(
                li, mode=mode, dtype=dtype
            )

            total_bytes = w_bytes + a_bytes + g_bytes + o_bytes
            ai = flops / total_bytes if total_bytes > 0 else 0.0
            attainable = hw.attainable_performance(ai, dtype)
            bottleneck = "compute" if ai > ridge else "memory"
            t_ms = flops / attainable * 1000.0 if attainable > 0 else 0.0

            layer_stats.append(
                LayerStats(
                    layer=li,
                    flops=flops,
                    weight_bytes=w_bytes,
                    activation_bytes=a_bytes,
                    grad_bytes=g_bytes,
                    optimizer_bytes=o_bytes,
                    ridge_point=ridge,
                    attainable_perf=attainable,
                    bottleneck=bottleneck,
                    theoretical_time_ms=t_ms,
                    hw_name=hw.name,
                    dtype_used=dtype,
                )
            )

        return AnalysisResults(layers=layer_stats, hw=hw, dtype=dtype, mode=mode)

    # ------------------------------------------------------------------

    def _parse(self, model, source, input_shapes, dtype) -> List[LayerInfo]:
        """Route to the appropriate parser based on source / model type."""
        source = source or _infer_source(model)

        if source == "pytorch":
            from roofline.parsers.pytorch_parser import PyTorchParser
            return PyTorchParser().parse(model, input_shapes=input_shapes, dtype=dtype)

        if source == "onnx":
            from roofline.parsers.onnx_parser import ONNXParser
            return ONNXParser().parse(model, input_shapes=input_shapes, dtype=dtype)

        if source == "huggingface":
            from roofline.parsers.huggingface_parser import HuggingFaceParser
            return HuggingFaceParser().parse(model, input_shapes=input_shapes, dtype=dtype)

        if source == "tensorflow":
            from roofline.parsers.tensorflow_parser import TensorFlowParser
            return TensorFlowParser().parse(model, input_shapes=input_shapes, dtype=dtype)

        if source == "folder":
            from roofline.parsers.folder_parser import FolderParser
            return FolderParser().parse(model, input_shapes=input_shapes, dtype=dtype)

        if source == "zip":
            from roofline.parsers.zip_handler import ZipHandler
            return ZipHandler().parse(model, input_shapes=input_shapes, dtype=dtype)

        raise ValueError(
            f"Unknown source '{source}'. Valid: pytorch, onnx, huggingface, "
            f"tensorflow, folder, zip."
        )


# ---------------------------------------------------------------------------
# Source auto-detection
# ---------------------------------------------------------------------------

def _infer_source(model) -> str:
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            return "pytorch"
    except ImportError:
        pass

    try:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
            return "tensorflow"
    except ImportError:
        pass

    if isinstance(model, str):
        p = Path(model)
        ext = p.suffix.lower()
        if ext == ".onnx":
            return "onnx"
        if ext in (".pt", ".pth"):
            return "pytorch"
        if ext in (".h5", ".keras"):
            return "tensorflow"
        if ext == ".zip":
            return "zip"
        if p.is_dir():
            return "folder"
        # Assume HuggingFace Hub model name
        return "huggingface"

    # onnx ModelProto
    try:
        import onnx
        if isinstance(model, onnx.ModelProto):
            return "onnx"
    except ImportError:
        pass

    raise TypeError(
        f"Cannot infer source for model of type {type(model).__name__}. "
        f"Pass source='pytorch'|'onnx'|'huggingface'|'tensorflow'|'folder'|'zip' explicitly."
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_flops(n: int) -> str:
    if n >= 1e12:
        return f"{n/1e12:.3f} TFLOPs"
    if n >= 1e9:
        return f"{n/1e9:.3f} GFLOPs"
    if n >= 1e6:
        return f"{n/1e6:.3f} MFLOPs"
    return f"{n} FLOPs"


def _fmt_params(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.3f} B"
    if n >= 1e6:
        return f"{n/1e6:.3f} M"
    if n >= 1e3:
        return f"{n/1e3:.1f} K"
    return str(n)


def _fmt_bytes(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f} GB"
    if n >= 1e6:
        return f"{n/1e6:.2f} MB"
    if n >= 1e3:
        return f"{n/1e3:.1f} KB"
    return f"{n} B"
