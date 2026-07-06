"""
Multi-Model × Multi-Hardware Roofline Analysis
===============================================
Runs roofline analysis across 4 torchvision models and 4 hardware targets,
producing:

  • 4 per-model plots  — one roofline figure per model showing all hardware
                          curves overlaid with annotated layer-wise scatter
  • 1 unified plot     — single-axes aggregate comparison (one dot per
                          model-hardware pair, no per-layer clutter)

Prerequisites
-------------
    pip install torch torchvision matplotlib

Usage
-----
    python examples/multi_model_multi_hw.py

Output files (written to the current working directory):
    alexnet_multi_hw_roofline.png
    mobilenetv2_multi_hw_roofline.png
    resnet101_multi_hw_roofline.png
    vgg16_multi_hw_roofline.png
    multi_model_multi_hw_roofline.png
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torchvision.models as tv_models

from roofline import analyze, HW_DB
from roofline.reporting.roofline_plot import plot_model_across_hw, plot_multi_model_multi_hw

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HW_TARGETS = [
    HW_DB["raspberry_pi4"],   # ARM Cortex-A72 | 8 GB LPDDR4 | 25.6 GB/s
    HW_DB["raspberry_pi5"],   # ARM Cortex-A76 | 8 GB LPDDR5 | 51.2 GB/s
    HW_DB["m4"],              # Apple M4        | 24 GB       | unified memory
    HW_DB["arduino_nicla"],   # ARM Cortex-M7   | 2 MB SRAM   | 15.36 GB/s
]

# Each entry: (torchvision constructor, input_shapes)
MODELS: dict = {
    "AlexNet":    (tv_models.alexnet,      [(1, 3, 224, 224)]),
    "MobileNetV2":(tv_models.mobilenet_v2, [(1, 3, 224, 224)]),
    "ResNet101":  (tv_models.resnet101,    [(1, 3, 224, 224)]),
    "VGG16":      (tv_models.vgg16,        [(1, 3, 224, 224)]),
}

DTYPE = "float32"
MODE  = "inference"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEP_WIDE  = "=" * 100
_SEP_THIN  = "-" * 100


def _fmt_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def _fmt_flops(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.1f}G"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return str(n)


def _print_hw_specs(hw) -> None:
    print(f"\n  [Hardware: {hw.name}]")
    print(f"    Architecture : {hw.architecture}")
    fp32_tflops = hw._get_peak_flops("float32") / 1e12
    fp16_tflops = hw._get_peak_flops("float16") / 1e12
    bw_gbs      = hw.peak_mem_bw / 1e9
    ridge       = hw.ridge_point(DTYPE)
    print(f"    Peak Compute : {fp32_tflops:.2f} TFLOPS (FP32) | {fp16_tflops:.2f} TFLOPS (FP16)")
    print(f"    Peak Mem BW  : {bw_gbs:.2f} GB/s  [{hw.memory_type}]")
    print(f"    Ridge Point  : {ridge:.2f} FLOPs/Byte")


def _print_layer_table(results) -> None:
    header = (
        f"{'Layer type':<20} {'Name':<32} {'Params':>10} "
        f"{'FLOPs':>10} {'AI':>8} {'Attain(T)':>10} {'Eff%':>6} {'Bottleneck'}"
    )
    print(f"\n{_SEP_THIN}")
    print("  Per-Layer Breakdown  (sorted by FLOPs desc)")
    print(_SEP_THIN)
    print(f"  {header}")
    print(f"  {_SEP_THIN}")
    for ls in sorted(results.layers, key=lambda x: x.flops, reverse=True):
        ltype  = ls.layer.layer_type[:19]
        name   = ls.layer.name.split(".")[-1][:31]
        params = _fmt_params(ls.layer.num_params)
        flops  = _fmt_flops(ls.flops)
        ai     = ls.arithmetic_intensity
        attain = ls.attainable_perf / 1e12
        eff    = ls.efficiency_pct()
        bot    = ls.bottleneck
        print(f"  {ltype:<20} {name:<32} {params:>10} {flops:>10} "
              f"{ai:>8.2f} {attain:>10.3f} {eff:>6.1f}  {bot}")
    print(f"  {_SEP_THIN}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(_SEP_WIDE)
    print("  Multi-Model × Multi-Hardware Roofline Analysis")
    print(f"  dtype={DTYPE}  mode={MODE}")
    print(_SEP_WIDE)

    # ---- Build results grid: {model_name: [AnalysisResults per HW]} --------
    results_grid: dict = {}

    for model_name, (model_fn, input_shapes) in MODELS.items():
        print(f"\n{'='*100}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*100}")

        model_obj = model_fn(weights=None)

        results_list = []
        for hw in HW_TARGETS:
            results = analyze(
                model=model_obj,
                input_shapes=input_shapes,
                hw=hw,
                dtype=DTYPE,
                mode=MODE,
            )
            results_list.append(results)

            _print_hw_specs(hw)
            results.print_summary()
            _print_layer_table(results)

        results_grid[model_name] = results_list

    # ---- Per-model plots (layer-wise scatter, all HW rooflines) -------------
    print(f"\n{_SEP_WIDE}")
    print("  Generating per-model roofline plots …")
    print(_SEP_WIDE)

    for model_name, results_list in results_grid.items():
        out_name = f"{model_name.lower().replace(' ', '_')}_multi_hw_roofline.png"
        plot_model_across_hw(
            model_name=model_name,
            results_list=results_list,
            save_path=out_name,
            show=False,          # set True to display interactively
            annotate_top_n=8,
        )

    # ---- Unified aggregate plot (one dot per model-HW pair) -----------------
    print(f"\n{_SEP_WIDE}")
    print("  Generating unified multi-model × multi-hardware plot …")
    print(_SEP_WIDE)

    plot_multi_model_multi_hw(
        results_grid=results_grid,
        save_path="multi_model_multi_hw_roofline.png",
        show=True,               # show the final combined figure
    )

    print(f"\n{_SEP_WIDE}")
    print("  Done.  Output files:")
    for model_name in results_grid:
        fname = f"{model_name.lower().replace(' ', '_')}_multi_hw_roofline.png"
        print(f"    {fname}")
    print("    multi_model_multi_hw_roofline.png")
    print(_SEP_WIDE)


if __name__ == "__main__":
    main()
