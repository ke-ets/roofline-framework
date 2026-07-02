"""
Roofline Analysis Framework
===========================
Layer-wise arithmetic intensity, FLOPs, and memory traffic estimation
for deep learning models, with hardware roofline analysis.

Quick start
-----------
>>> from roofline import analyze, HW_DB
>>> results = analyze(model, input_shapes=[(1, 3, 224, 224)], hw=HW_DB["h100_sxm"])
>>> results.print_table()
>>> results.plot_roofline()
"""

from roofline.core.analyzer import RooflineAnalyzer, AnalysisResults
from roofline.hardware.hw_database import HW_DB
from roofline.hardware.hw_detector import detect_hw
from roofline.hardware.hw_fetcher import fetch_hw
from roofline.hardware.hw_resolver import HWResolver

__all__ = [
    "analyze",
    "RooflineAnalyzer",
    "AnalysisResults",
    "HW_DB",
    "detect_hw",
    "fetch_hw",
    "HWResolver",
]


def analyze(
    model,
    input_shapes=None,
    hw=None,
    dtype: str = "float32",
    mode: str = "inference",
    source: str = None,
    batch_size: int = None,
    fetch_from_web: bool = False,
    detect_local_hw: bool = False,
):
    """Perform roofline analysis on a model.

    Parameters
    ----------
    model:
        The model to analyse. Accepts:
        - ``torch.nn.Module`` instance
        - ``tf.keras.Model`` instance
        - Path string to an ``.onnx``, ``.pt``, ``.h5``, ``.zip`` file
        - Path string to a directory (HuggingFace local save or manual layer files)
        - HuggingFace Hub model name (e.g. ``"bert-base-uncased"``)
    input_shapes:
        List of (batch, ...) tuples for each model input.
        Required for PyTorch / ONNX; inferred for HuggingFace / Keras where possible.
    hw:
        ``HWSpec`` instance from ``HW_DB`` or returned by ``detect_hw()`` / ``fetch_hw()``.
        If None and ``detect_local_hw=True``, auto-detected from the current machine.
    dtype:
        One of ``"float32"``, ``"float16"``, ``"bfloat16"``, ``"int8"``, ``"int4"``.
    mode:
        ``"inference"`` (weights + activations) or ``"training"``
        (weights + activations + gradients + optimizer states).
    source:
        Explicit source override: ``"pytorch"``, ``"onnx"``, ``"huggingface"``,
        ``"tensorflow"``, ``"folder"``, ``"zip"``.  Auto-detected when omitted.
    batch_size:
        Override batch dimension in all input shapes.
    fetch_from_web:
        Allow fetching unknown HW specs from TechPowerUp without a CLI prompt.
    detect_local_hw:
        Auto-detect the GPU/CPU this machine is running on when ``hw`` is None.

    Returns
    -------
    AnalysisResults
    """
    analyzer = RooflineAnalyzer()
    return analyzer.analyze(
        model=model,
        input_shapes=input_shapes,
        hw=hw,
        dtype=dtype,
        mode=mode,
        source=source,
        batch_size=batch_size,
        fetch_from_web=fetch_from_web,
        detect_local_hw=detect_local_hw,
    )
