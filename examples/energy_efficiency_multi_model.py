"""
Energy Efficiency — Multi-Model × Multi-Hardware Roofline Analysis
==================================================================
Runs roofline + energy analysis across 4 torchvision models and 4 hardware
targets, producing:

  • 4 per-model dual-axis plots  — perf roofline (left) + FLOPS/J curve (right)
  • 1 combined dual-axis plot    — all models × all HW on a single figure

Prerequisites
-------------
    pip install torch torchvision matplotlib

Usage
-----
    python examples/energy_efficiency_multi_model.py

Output files (written to the current working directory):
    alexnet_energy_roofline.png
    mobilenet_v2_energy_roofline.png
    resnet101_energy_roofline.png
    vgg16_energy_roofline.png
    multi_model_energy_roofline.png
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torchvision.models as tv_models

from roofline import analyze, HW_DB
from roofline.reporting.roofline_plot import (
    plot_model_across_hw_energy,
    plot_multi_model_multi_hw_energy,
)
from roofline.reporting.table_report import TableReport

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HW_TARGETS = [
    HW_DB["raspberry_pi4"],
    HW_DB["raspberry_pi5"],
    HW_DB["arduino_nicla"],
    HW_DB["m4"],
]

MODELS: dict = {
    "alexnet":      (tv_models.alexnet,      [(1, 3, 224, 224)]),
    "mobilenet_v2": (tv_models.mobilenet_v2, [(1, 3, 224, 224)]),
    "resnet101":    (tv_models.resnet101,    [(1, 3, 224, 224)]),
    "vgg16":        (tv_models.vgg16,        [(1, 3, 224, 224)]),
}

DTYPE = "float32"
MODE  = "inference"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEP_WIDE = "=" * 100
_SEP_THIN = "-" * 100


def _fmt_flops(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}G"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return str(n)


def _print_model_hw_summary(model_name: str, results) -> None:
    """Print the per-HW summary including energy metrics."""
    hw = results.hw
    print(f"\n  [{hw.name}]")
    print(results.summary())


def _print_energy_layer_table(results) -> None:
    """Print a compact per-layer table showing energy columns."""
    header = (
        f"{'Layer type':<20} {'Name':<30} {'FLOPs':>10} {'AI':>8} "
        f"{'Bottleneck':<12} {'Energy(µJ)':>12} {'GFLOPS/J':>10}"
    )
    print(f"\n{_SEP_THIN}")
    print("  Per-Layer Energy Breakdown  (sorted by energy desc)")
    print(_SEP_THIN)
    print(f"  {header}")
    print(f"  {_SEP_THIN}")
    for ls in sorted(results.layers, key=lambda x: x.energy_j, reverse=True):
        ltype  = ls.layer.layer_type[:19]
        name   = ls.layer.name.split(".")[-1][:29]
        flops  = _fmt_flops(ls.flops)
        ai     = ls.arithmetic_intensity
        bot    = ls.bottleneck
        e_uj   = ls.energy_j * 1e6
        e_eff  = ls.energy_efficiency / 1e9
        print(f"  {ltype:<20} {name:<30} {flops:>10} {ai:>8.2f} "
              f"{bot:<12} {e_uj:>12.4f} {e_eff:>10.4f}")
    print(f"  {_SEP_THIN}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(_SEP_WIDE)
    print("  Energy Efficiency — Multi-Model × Multi-Hardware Roofline Analysis")
    print(f"  dtype={DTYPE}  mode={MODE}")
    print(_SEP_WIDE)

    # ---- 1. Build results grid: {model_name: [AnalysisResults per HW]} ----
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

        results_grid[model_name] = results_list

    # ---- 2. Print per-HW summary (with energy) for each model --------------
    print(f"\n{_SEP_WIDE}")
    print("  Per-HW Summary (including total_energy_j and energy_efficiency)")
    print(_SEP_WIDE)

    for model_name, results_list in results_grid.items():
        print(f"\n{'='*80}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*80}")
        for results in results_list:
            _print_model_hw_summary(model_name, results)

    # ---- 3. Print per-layer energy table for each (model, HW) pair ---------
    print(f"\n{_SEP_WIDE}")
    print("  Per-Layer Energy Breakdown")
    print(_SEP_WIDE)

    for model_name, results_list in results_grid.items():
        for results in results_list:
            print(f"\n{'='*80}")
            print(f"  {model_name}  on  {results.hw.name}")
            print(f"{'='*80}")
            _print_energy_layer_table(results)

    # ---- 4. Per-model dual-axis energy roofline plots ----------------------
    print(f"\n{_SEP_WIDE}")
    print("  Generating per-model energy roofline plots …")
    print(_SEP_WIDE)

    for model_name, results_list in results_grid.items():
        out_name = f"{model_name}_energy_roofline.png"
        plot_model_across_hw_energy(
            model_name=model_name,
            results_list=results_list,
            save_path=out_name,
            show=False,
        )

    # ---- 5. Combined multi-model × multi-HW energy plot --------------------
    print(f"\n{_SEP_WIDE}")
    print("  Generating combined multi-model × multi-hardware energy plot …")
    print(_SEP_WIDE)

    plot_multi_model_multi_hw_energy(
        results_grid=results_grid,
        save_path="multi_model_energy_roofline.png",
        show=True,
    )

    # ---- 6. Final summary table: one row per (model, HW) -------------------
    print(f"\n{_SEP_WIDE}")
    print("  Final Summary Table")
    print(_SEP_WIDE)

    col_w = [16, 22, 14, 16, 20]
    header = (
        f"{'Model':<{col_w[0]}} {'Hardware':<{col_w[1]}} "
        f"{'Total FLOPs':>{col_w[2]}} {'Energy (µJ)':>{col_w[3]}} "
        f"{'Effic. (GFLOPS/J)':>{col_w[4]}}"
    )
    print(f"\n  {header}")
    print(f"  {'-' * sum(col_w)}")

    for model_name, results_list in results_grid.items():
        for results in results_list:
            total_flops   = results.total_flops
            total_energy  = results.total_energy_j * 1e6    # µJ
            effic         = results.energy_efficiency / 1e9  # GFLOPS/J
            hw_name       = results.hw.name
            flops_str     = _fmt_flops(total_flops)
            print(
                f"  {model_name:<{col_w[0]}} {hw_name:<{col_w[1]}} "
                f"{flops_str:>{col_w[2]}} {total_energy:>{col_w[3]}.4f} "
                f"{effic:>{col_w[4]}.4f}"
            )

    print(f"\n{_SEP_WIDE}")
    print("  Done.  Output files:")
    for model_name in results_grid:
        print(f"    {model_name}_energy_roofline.png")
    print("    multi_model_energy_roofline.png")
    print(_SEP_WIDE)


if __name__ == "__main__":
    main()
