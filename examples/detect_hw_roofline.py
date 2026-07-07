"""
HW-Detect Roofline Experiment
==============================
Auto-detects the local hardware (NVIDIA GPU → AMD GPU → Apple Silicon → CPU),
then runs roofline analysis on 7 models spanning torchvision CNNs and
HuggingFace language models.

Outputs
-------
Per-model layer-wise plots (7 files):
    alexnet_roofline.png
    resnet101_roofline.png
    vgg16_roofline.png
    mobilenetv2_roofline.png
    gpt-2_roofline.png
    deepseek-1.3b_roofline.png
    gemma-2b_roofline.png   (skipped if HF token not set up)

Consolidated aggregate plot (1 file):
    all_models_on_detected_hw.png

Usage
-----
    python examples/detect_hw_roofline.py

All missing dependencies (torchvision, transformers, huggingface_hub) are
installed automatically on first run.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Auto-install missing dependencies before any other import
# ---------------------------------------------------------------------------
import importlib.util
import subprocess
import sys


def _ensure(package: str, pip_name: str = None) -> None:
    """pip-install `pip_name` if `package` cannot be imported."""
    if importlib.util.find_spec(package) is None:
        target = pip_name or package
        print(f"[setup] Installing missing dependency: {target} ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "--trusted-host", "pypi.org",
             "--trusted-host", "files.pythonhosted.org",
             "--trusted-host", "pypi.python.org",
             target]
        )
        print(f"[setup] {target} installed.")


_ensure("torchvision")
_ensure("transformers")
_ensure("huggingface_hub", "huggingface_hub")

# ---------------------------------------------------------------------------
# Standard imports (after deps are guaranteed)
# ---------------------------------------------------------------------------
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torchvision.models as tv_models

from roofline import analyze, HW_DB, detect_hw, fetch_hw
from roofline.hardware.hw_detector import HWDetectionError
from roofline.reporting.roofline_plot import plot_models_on_hw

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DTYPE = "float32"
MODE  = "inference"

# Torchvision models: (constructor_fn, input_shapes)
_TV_MODELS = {
    "AlexNet":    (lambda: tv_models.alexnet(weights=None),      [(1, 3, 224, 224)]),
    "ResNet101":  (lambda: tv_models.resnet101(weights=None),    [(1, 3, 224, 224)]),
    "VGG16":      (lambda: tv_models.vgg16(weights=None),        [(1, 3, 224, 224)]),
    "MobileNetV2":(lambda: tv_models.mobilenet_v2(weights=None), [(1, 3, 224, 224)]),
}

# HuggingFace models: (hub_id, source_tag)
# Input shapes are inferred automatically from model config.
_HF_MODELS = {
    "GPT-2":        "openai-community/gpt2",
    "DeepSeek-1.3B":"deepseek-ai/deepseek-coder-1.3b-base",
    "Gemma-2B":     "google/gemma-2b",
}

_SEP = "=" * 100
_THIN = "-" * 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_flops(n: int) -> str:
    if n >= 1e12: return f"{n/1e12:.3f} TFLOPs"
    if n >= 1e9:  return f"{n/1e9:.3f} GFLOPs"
    if n >= 1e6:  return f"{n/1e6:.3f} MFLOPs"
    return f"{n} FLOPs"


def _fmt_params(n: int) -> str:
    if n >= 1e9: return f"{n/1e9:.2f}B"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.1f}K"
    return str(n)


def _print_summary(model_name: str, results) -> None:
    hw = results.hw
    print(f"\n{_SEP}")
    print(f"  MODEL: {model_name}  |  HW: {hw.name}  |  {results.dtype}  |  {results.mode}")
    print(_SEP)
    results.print_summary()
    print()
    print(f"  {_THIN}")
    print(f"  {'Layer type':<20} {'Name':<30} {'Params':>10} {'FLOPs':>10} "
          f"{'AI':>8} {'Attain(T)':>10} {'Eff%':>6}  Bottleneck")
    print(f"  {_THIN}")
    for ls in sorted(results.layers, key=lambda x: x.flops, reverse=True):
        ltype  = ls.layer.layer_type[:19]
        name   = ls.layer.name.split(".")[-1][:29]
        params = _fmt_params(ls.layer.num_params)
        flops  = (_fmt_flops(ls.flops).replace(" FLOPs", "")
                  .replace("FLOPs", "").strip())
        ai     = ls.arithmetic_intensity
        attain = ls.attainable_perf / 1e12
        eff    = ls.efficiency_pct()
        bot    = ls.bottleneck
        print(f"  {ltype:<20} {name:<30} {params:>10} {flops:>10} "
              f"{ai:>8.2f} {attain:>10.3f} {eff:>6.1f}  {bot}")
    print(f"  {_THIN}\n")


def _detect_hardware():
    """Detect local hardware with graceful fallback."""
    print(f"\n{_SEP}")
    print("  Hardware Detection")
    print(_SEP)

    try:
        hw = detect_hw()
        print(f"  [OK] Detected and matched: {hw.name}")
        print(f"       Architecture : {hw.architecture}")
        print(f"       Peak FP32    : {hw._get_peak_flops('float32')/1e12:.2f} TFLOPS")
        print(f"       Peak Mem BW  : {hw.peak_mem_bw/1e9:.2f} GB/s  [{hw.memory_type}]")
        print(f"       Ridge Point  : {hw.ridge_point(DTYPE):.2f} FLOPs/Byte")
        return hw
    except HWDetectionError as e:
        # Extract the raw detected name from the error message
        msg = str(e)
        try:
            detected_name = msg.split("'")[1]
        except IndexError:
            detected_name = "unknown"

        print(f"  [warn] Detected '{detected_name}' — not in built-in HW_DB.")
        print("  Attempting to fetch specs from TechPowerUp ...")
        try:
            hw = fetch_hw(detected_name, fetch_from_web=True)
            print(f"  [OK] Fetched: {hw.name}")
            return hw
        except Exception as fetch_err:
            print(f"  [warn] Web fetch failed: {fetch_err}")
            fallback = HW_DB["raspberry_pi5"]
            print(f"  [fallback] Using '{fallback.name}' as reference hardware.")
            print("             Edit the script and set hw = HW_DB['<key>'] to override.")
            return fallback


def _analyze_torchvision(hw) -> dict:
    """Run analysis on the 4 torchvision CNN models."""
    results = {}
    for model_name, (model_fn, input_shapes) in _TV_MODELS.items():
        print(f"\n  Analyzing {model_name} ...")
        try:
            model_obj = model_fn()
            r = analyze(model=model_obj, input_shapes=input_shapes,
                        hw=hw, dtype=DTYPE, mode=MODE)
            results[model_name] = r
            _print_summary(model_name, r)
        except Exception as e:
            print(f"  [skip] {model_name} failed: {e}")
    return results


def _analyze_huggingface(hw) -> dict:
    """Run analysis on the 3 HuggingFace language models."""
    results = {}
    for model_name, hub_id in _HF_MODELS.items():
        print(f"\n  Analyzing {model_name} ({hub_id}) ...")
        try:
            r = analyze(model=hub_id, hw=hw, dtype=DTYPE, mode=MODE,
                        source="huggingface")
            results[model_name] = r
            _print_summary(model_name, r)
        except OSError as e:
            # Likely a HuggingFace auth / token error (Gemma requires login)
            if "token" in str(e).lower() or "gated" in str(e).lower() or "401" in str(e):
                print(f"  [skip] {model_name}: HuggingFace authentication required.")
                print(f"         Run `huggingface-cli login` and re-run the script.")
            else:
                print(f"  [skip] {model_name}: {e}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [skip] {model_name}: Out of memory — try a smaller variant.")
            else:
                print(f"  [skip] {model_name}: {e}")
        except Exception as e:
            print(f"  [skip] {model_name}: {e}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(_SEP)
    print("  HW-Detect Roofline Experiment")
    print(f"  dtype={DTYPE}  mode={MODE}")
    print(_SEP)

    # 1. Detect hardware
    hw = _detect_hardware()

    # 2. Run analyses
    print(f"\n{_SEP}")
    print("  Analyzing torchvision CNN models ...")
    print(_SEP)
    tv_results = _analyze_torchvision(hw)

    print(f"\n{_SEP}")
    print("  Analyzing HuggingFace language models ...")
    print(_SEP)
    hf_results = _analyze_huggingface(hw)

    results_dict = {**tv_results, **hf_results}

    if not results_dict:
        print("\n[error] No models were successfully analyzed. Exiting.")
        return

    # 3. Per-model layer-wise roofline plots
    print(f"\n{_SEP}")
    print("  Generating per-model layer-wise roofline plots ...")
    print(_SEP)
    for model_name, results in results_dict.items():
        fname = f"{model_name.lower().replace(' ', '_')}_roofline.png"
        results.plot_roofline(save_path=fname, show=False, annotate_top_n=5,
                              model_name=model_name)

    # 4. Consolidated aggregate plot
    print(f"\n{_SEP}")
    print("  Generating consolidated model-vs-hardware plot ...")
    print(_SEP)
    plot_models_on_hw(
        results_dict=results_dict,
        save_path="all_models_on_detected_hw.png",
        show=True,
    )

    # 5. Summary
    print(f"\n{_SEP}")
    print(f"  Done.  Hardware used: {hw.name}")
    print(f"  Output files:")
    for model_name in results_dict:
        print(f"    {model_name.lower().replace(' ', '_')}_roofline.png")
    print("    all_models_on_detected_hw.png")
    print(_SEP)


if __name__ == "__main__":
    main()
