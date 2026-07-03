"""
AlexNet Roofline Analysis Example
==================================
Analyses a locally-saved AlexNet model (individual per-layer .pt tensor files)
and produces a summary table + roofline plot.

Usage
-----
    python examples/alexnet_example.py

Point MODEL_FOLDER at the directory that contains the .pt weight/bias files,
then choose a hardware target from HW_DB (run `python -m roofline.cli list-hw`
to see all available targets).
"""

import sys
import os

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roofline import analyze, HW_DB
from roofline.reporting.roofline_plot import plot_multiple

# -----------------------------------------------------------------
# Configuration — edit these values to match your setup
# -----------------------------------------------------------------
MODEL_FOLDER = r"C:\Users\keeg\OneDrive - Nokia\my_inno_work\koold\edgeAI\ROOFLINE\base_alexnet_wb\base_alexnet_wb"
HW_TARGETS = [
    HW_DB["m4"],
    HW_DB["raspberry_pi5"],
    HW_DB["raspberry_pi4"],
    HW_DB["arduino_nicla"],
]
DTYPE = "float32"
MODE = "inference"
# -----------------------------------------------------------------

results_list = [
    analyze(model=MODEL_FOLDER, hw=hw, dtype=DTYPE, mode=MODE)
    for hw in HW_TARGETS
]

# 1. One-line summary, hardware details, and per-layer tables for each target
for results in results_list:
    hw = results.hw
    print(f"\n{'='*90}")
    print(f"ANALYSIS: {hw.name} | {results.dtype} | {results.mode}")
    print(f"{'='*90}")

    peak_fp32 = hw._get_peak_flops("float32") / 1e12
    peak_fp16 = hw._get_peak_flops("float16") / 1e12
    peak_bw_gbs = hw.peak_mem_bw / 1e9
    ridge = hw.ridge_point(results.dtype)

    print("\n[Hardware Specs]")
    print(f"  Architecture    : {hw.architecture}")
    print(f"  Peak Compute    : {peak_fp32:.2f} TFLOPS (FP32) | {peak_fp16:.2f} TFLOPS (FP16)")
    print(f"  Peak Mem BW     : {peak_bw_gbs:.2f} GB/s")
    print(f"  Memory Type     : {hw.memory_type}")
    print(f"  Ridge Point     : {ridge:.1f} FLOPs/Byte")
    print()

    results.print_summary()
    print()

    print(f"\n{'-'*90}")
    print("Layer-by-Layer Breakdown:")
    print(f"{'-'*90}")
    print(f"{'Type':<20} {'Params':>12} {'FLOPs':>12} {'AI':>8} {'Attain':>10} {'Eff%':>6} {'Bottleneck':<10}")
    print(f"{'-'*90}")

    for layer_stat in sorted(results.layers, key=lambda x: x.flops, reverse=True):
        layer_type = layer_stat.layer.layer_type[:19]
        params = layer_stat.layer.num_params
        flops = layer_stat.flops
        ai = layer_stat.arithmetic_intensity
        attain_tflops = layer_stat.attainable_perf / 1e12
        eff_pct = layer_stat.efficiency_pct()
        bottleneck = layer_stat.bottleneck

        params_str = f"{params/1e6:.1f}M" if params >= 1e6 else f"{params/1e3:.1f}K" if params >= 1e3 else str(params)
        flops_str = f"{flops/1e9:.1f}G" if flops >= 1e9 else f"{flops/1e6:.1f}M" if flops >= 1e6 else str(flops)

        print(f"{layer_type:<20} {params_str:>12} {flops_str:>12} {ai:>8.2f} {attain_tflops:>10.2f} {eff_pct:>6.1f} {bottleneck:<10}")

    print(f"{'-'*90}\n")
    results.print_table()

# 2. Combined roofline comparison plot for all targets
plot_multiple(
    results_list,
    save_path="alexnet_multi_hw_roofline.png",
    show=True,
)

# 3. DataFrame preview for each target
for results in results_list:
    df = results.to_dataframe()
    print(f"\n{'='*90}")
    print(f"DataFrame preview for {results.hw.name}")
    print(f"{'='*90}")
    print(df[["type", "params", "flops", "arithmetic_intensity", "efficiency_pct", "bottleneck"]].head(10))
