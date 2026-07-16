"""
Time Analysis Roofline Demo
===========================
Extends roofline analysis with timing metrics (T_compute, T_memory, T_actual)
for 4 models × 4 hardware targets, producing 3 sets of plots:

  Set 1  (4 figures):  Per-model dual-axis plot — all 4 HW overlaid.
                        Left Y  = Attainable Perf (FLOPs/s)  [solid rooflines]
                        Right Y = T_actual per layer  (log seconds)  [dots]
                        Top-10 unique layer types shown.
                        Output: time_set1_{model}.png

  Set 2  (1 figure):   All models × all HW aggregate timing on one plot.
                        Left Y  = Attainable Perf  [solid rooflines]
                        Right Y = T_actual per model-HW aggregate dot
                        Output: time_set2_aggregate.png

  Set 3  (16 figures): Per-model per-HW layerwise bar chart.
                        Left Y  = Attainable Perf  [solid roofline + reference dots]
                        Right Y = T_memory / T_compute adjacent bars
                        Top-5 instances per layer_type; layer names below X axis.
                        Output: time_set3_{hw}_{model}.png

Key Formulas
------------
    Y         = min(peak_perf, I * peak_mem_bw)   attainable performance
    T_compute = W / Y                              compute-bound ideal time
    T_memory  = Q / peak_mem_bw                   memory-bound ideal time
    T_actual  = max(T_compute, T_memory)           bottleneck-determined time

Prerequisites
-------------
    pip install torch torchvision matplotlib

Usage
-----
    cd roofline-framework
    python examples/time_analysis_demo.py

Output PNGs are written to the current working directory.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import torchvision.models as tv_models

from roofline import analyze, HW_DB

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HW_TARGETS = [
    HW_DB["raspberry_pi4"],    # ARM Cortex-A72 | LPDDR4 25.6 GB/s
    HW_DB["raspberry_pi5"],    # ARM Cortex-A76 | LPDDR5 51.2 GB/s
    HW_DB["m4"],               # Apple M4       | LPDDR5X 120 GB/s
    HW_DB["arduino_nicla"],    # ARM Cortex-M7  | SRAM 15.36 GB/s
]

MODELS: dict = {
    "AlexNet":    (tv_models.alexnet,      [(1, 3, 224, 224)]),
    "MobileNetV2":(tv_models.mobilenet_v2, [(1, 3, 224, 224)]),
    "ResNet101":  (tv_models.resnet101,    [(1, 3, 224, 224)]),
    "VGG16":      (tv_models.vgg16,        [(1, 3, 224, 224)]),
}

DTYPE = "float32"
MODE  = "inference"

# Marker shapes: HW for Set 1, Model for Set 2
_HW_MARKERS    = ["o", "s", "^", "D"]
_MODEL_MARKERS = ["o", "s", "^", "D"]

# Layer type → colour (mirrors roofline_plot.py for visual consistency)
_TYPE_COLORS: dict = {
    "Linear":           "#4C72B0",
    "MatMul":           "#4C72B0",
    "Gemm":             "#4C72B0",
    "Conv1d":           "#DD8452",
    "Conv2d":           "#DD8452",
    "Conv3d":           "#DD8452",
    "ConvTranspose1d":  "#DD8452",
    "ConvTranspose2d":  "#DD8452",
    "ConvTranspose3d":  "#DD8452",
    "MultiHeadAttention":"#55A868",
    "Attention":        "#55A868",
    "LSTM":             "#C44E52",
    "GRU":              "#C44E52",
    "RNN":              "#C44E52",
    "BatchNorm":        "#8172B3",
    "LayerNorm":        "#8172B3",
    "GroupNorm":        "#8172B3",
    "InstanceNorm":     "#8172B3",
    "Embedding":        "#937860",
    "ReLU":             "#DA8BC3",
    "GELU":             "#DA8BC3",
    "SiLU":             "#DA8BC3",
    "Activation":       "#DA8BC3",
    "MaxPool":          "#3A86FF",
    "AvgPool":          "#06D6A0",
    "AdaptiveAvgPool":  "#06D6A0",
    "Elementwise":      "#CCB974",
    "Reshape":          "#64B5CD",
    "Dropout":          "#AAAAAA",
}
_DEFAULT_COLOR = "#888888"

# Bar colours for Set 3
_BAR_COLOR_MEM  = "#4C9BE8"   # blue  → T_memory
_BAR_COLOR_COMP = "#F4845F"   # orange → T_compute


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_time(t_s: float) -> str:
    """Format a time in seconds to a human-readable string."""
    if t_s <= 0:
        return "0 s"
    if t_s >= 1.0:
        return f"{t_s:.3f} s"
    if t_s >= 1e-3:
        return f"{t_s * 1e3:.3f} ms"
    if t_s >= 1e-6:
        return f"{t_s * 1e6:.3f} us"
    return f"{t_s * 1e9:.2f} ns"


def _fmt_flops(n: int | float) -> str:
    if n >= 1e12: return f"{n / 1e12:.2f}T"
    if n >= 1e9:  return f"{n / 1e9:.2f}G"
    if n >= 1e6:  return f"{n / 1e6:.1f}M"
    if n >= 1e3:  return f"{n / 1e3:.0f}K"
    return str(int(n))


def _time_fmt_axis(v: float, _) -> str:
    """Formatter for a log-scaled time axis."""
    if v <= 0:
        return ""
    if v < 1e-6:
        return f"{v * 1e9:.0f}ns"
    if v < 1e-3:
        return f"{v * 1e6:.0f}us"
    if v < 1.0:
        return f"{v * 1e3:.1f}ms"
    return f"{v:.2f}s"


def _perf_fmt_axis(v: float, _) -> str:
    """Formatter for a log-scaled performance axis."""
    if v >= 1e12:
        return f"{v / 1e12:.0f}T"
    if v >= 1e9:
        return f"{v / 1e9:.0f}G"
    if v >= 1e6:
        return f"{v / 1e6:.0f}M"
    return f"{v:.0e}"


# ---------------------------------------------------------------------------
# Core: time metric computation
# ---------------------------------------------------------------------------

def compute_time_metrics(results) -> dict:
    """Compute T_compute, T_memory, T_actual for every layer and the whole model.

    Returns
    -------
    dict with keys:
        "layers"  : list of per-layer metric dicts
        "model"   : model-aggregate metric dict
        "hw"      : HWSpec used
    """
    hw       = results.hw
    peak_bw  = hw.peak_mem_bw
    dtype    = results.dtype

    layers_out: list = []
    for ls in results.layers:
        if ls.flops == 0 and ls.total_bytes == 0:
            continue
        W = ls.flops
        Q = ls.total_bytes
        I = ls.arithmetic_intensity
        Y = ls.attainable_perf                          # min(I*bw, peak_flops)
        T_compute = ls.theoretical_time_ms / 1000.0    # seconds; W/Y already computed
        T_memory  = Q / peak_bw if peak_bw > 0 else 0.0
        T_actual  = max(T_compute, T_memory)
        layers_out.append({
            "name":       ls.layer.name,
            "layer_type": ls.layer.layer_type,
            "W":          W,
            "Q":          Q,
            "I":          I,
            "Y":          Y,
            "T_compute":  T_compute,
            "T_memory":   T_memory,
            "T_actual":   T_actual,
            "bottleneck": ls.bottleneck,
        })

    # Model-level aggregate
    W_total    = results.total_flops
    Q_total    = results.total_bytes
    I_model    = W_total / Q_total if Q_total > 0 else 0.0
    peak_flops = hw._get_peak_flops(dtype)
    Y_model    = min(I_model * peak_bw, peak_flops) if I_model > 0 else 0.0
    T_compute_model = W_total / Y_model if Y_model > 0 else 0.0
    T_memory_model  = Q_total / peak_bw if peak_bw > 0 else 0.0
    T_actual_model  = max(T_compute_model, T_memory_model)

    return {
        "layers": layers_out,
        "model": {
            "W_total":         W_total,
            "Q_total":         Q_total,
            "I_model":         I_model,
            "Y_model":         Y_model,
            "T_compute_model": T_compute_model,
            "T_memory_model":  T_memory_model,
            "T_actual_model":  T_actual_model,
        },
        "hw": hw,
    }


# ---------------------------------------------------------------------------
# Layer selection helpers
# ---------------------------------------------------------------------------

def select_top10_unique_layers(layer_metrics: list) -> list:
    """One best representative per layer_type, top-10 types by W (FLOPs)."""
    type_rep: dict = {}
    for lm in layer_metrics:
        lt = lm["layer_type"]
        if lt in ("Unknown", "Dropout", "Reshape"):
            continue
        if lt not in type_rep or lm["W"] > type_rep[lt]["W"]:
            type_rep[lt] = lm
    ranked = sorted(type_rep.values(), key=lambda x: x["W"], reverse=True)
    return ranked[:10]


def select_top5_per_type(layer_metrics: list) -> list:
    """Top-5 instances per layer_type by W; result ordered by AI ascending."""
    groups: dict = defaultdict(list)
    for lm in layer_metrics:
        lt = lm["layer_type"]
        if lt in ("Unknown", "Dropout", "Reshape") or lm["W"] == 0:
            continue
        groups[lt].append(lm)
    selected: list = []
    for lt, items in groups.items():
        top5 = sorted(items, key=lambda x: x["W"], reverse=True)[:5]
        selected.extend(top5)
    selected.sort(key=lambda x: x["I"])   # left-to-right = memory → compute bound
    return selected


# ---------------------------------------------------------------------------
# Shared roofline x-axis range helper
# ---------------------------------------------------------------------------

def _x_range(results_list: list, metrics_list: list, dtype: str):
    all_ai = [max(lm["I"], 1e-3)
              for m in metrics_list for lm in m["layers"] if lm["I"] > 0]
    all_ridges = [r.hw.ridge_point(dtype) for r in results_list]
    ai_min = min(min(all_ai) if all_ai else 1e-2, 1e-2)
    ai_max = max(max(all_ridges) * 10, max(all_ai) * 10 if all_ai else 1e4, 1e4)
    return np.logspace(np.log10(ai_min), np.log10(ai_max), 500)


# ---------------------------------------------------------------------------
# Plot Set 1 — Per-model, all HW overlaid, top-10 unique layer dots
# ---------------------------------------------------------------------------

def plot_set1_model_time(
    model_name: str,
    results_list: list,
    metrics_list: list,
    save_path: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """Dual-axis roofline + T_actual scatter for one model across all hardware.

    Left Y  (solid curves): Attainable Performance (FLOPs/s).
    Right Y (scatter dots): T_actual per layer (seconds, log scale).
    """
    first_result = results_list[0]
    dtype        = first_result.dtype
    mode         = first_result.mode
    n_layers     = len(first_result.layers)

    x_curve = _x_range(results_list, metrics_list, dtype)
    cmap    = plt.get_cmap("tab10")

    fig, ax_left = plt.subplots(figsize=(15, 8))
    ax_right = ax_left.twinx()

    # Top-10 unique layer types from the reference (first) HW
    ref_top10      = select_top10_unique_layers(metrics_list[0]["layers"])
    top10_names    = {lm["name"] for lm in ref_top10}

    ref_known:   list = []
    ref_ai_vals: list = []
    ref_t_vals:  list = []

    for idx, (result, metrics) in enumerate(zip(results_list, metrics_list)):
        hw         = result.hw
        color      = cmap(idx % 10)
        marker     = _HW_MARKERS[idx % len(_HW_MARKERS)]
        peak_flops = hw._get_peak_flops(dtype)
        peak_bw    = hw.peak_mem_bw

        # Left axis — solid roofline ceiling
        y_perf = np.minimum(x_curve * peak_bw, peak_flops)
        ax_left.plot(x_curve, y_perf, color=color, linewidth=2.2,
                     linestyle="-", zorder=5)

        # Right axis — T_actual dots for the top-10 layer set
        known = [lm for lm in metrics["layers"]
                 if lm["name"] in top10_names
                 and lm["I"] > 0 and lm["T_actual"] > 0]
        if not known:
            continue

        ai_vals  = [max(lm["I"], 1e-3) for lm in known]
        t_vals   = [lm["T_actual"] for lm in known]
        dot_cols = [_TYPE_COLORS.get(lm["layer_type"], _DEFAULT_COLOR) for lm in known]

        ax_right.scatter(ai_vals, t_vals,
                         marker=marker, c=dot_cols, s=90, alpha=0.88,
                         edgecolors=color, linewidths=1.2, zorder=12)

        if idx == 0:
            ref_known   = known
            ref_ai_vals = ai_vals
            ref_t_vals  = t_vals

    # Numbered labels on reference-HW dots + legend table
    if ref_known:
        sort_idx = sorted(range(len(ref_known)),
                          key=lambda i: ref_known[i]["W"], reverse=True)
        legend_lines: list = []
        for rank, i in enumerate(sort_idx):
            num = rank + 1
            ax_right.text(ref_ai_vals[i], ref_t_vals[i], str(num),
                          ha="center", va="center",
                          fontsize=7, fontweight="bold", color="white", zorder=25)
            lm    = ref_known[i]
            short = lm["name"].split("/")[-1]
            short = short[:36] + "..." if len(short) > 36 else short
            legend_lines.append(
                f"{num:>2}. [{lm['layer_type']:<14}]  {short:<36}"
                f"  ({_fmt_flops(lm['W'])} FLOPs)"
            )

        legend_text = "Top layers by FLOPs  [ref: " + results_list[0].hw.name + "]:\n" \
                      + "\n".join(legend_lines)
        ax_left.text(
            0.99, 0.02, legend_text,
            transform=ax_left.transAxes,
            fontsize=6.5, verticalalignment="bottom", horizontalalignment="right",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="#cccccc", alpha=0.92),
            zorder=20,
        )

    # Axis formatting
    ax_left.set_xscale("log")
    ax_left.set_yscale("log")
    ax_right.set_yscale("log")

    ax_left.set_xlabel("Arithmetic Intensity (FLOPs/Byte)", fontsize=12)
    ax_left.set_ylabel("Attainable Performance (FLOPs/s)", fontsize=12)
    ax_right.set_ylabel("T_actual — Layer Inference Time  (seconds, log)", fontsize=12)

    ax_left.yaxis.set_major_formatter(plt.FuncFormatter(_perf_fmt_axis))
    ax_right.yaxis.set_major_formatter(plt.FuncFormatter(_time_fmt_axis))

    ax_left.set_title(
        f"Time Analysis — {model_name} across Hardware\n"
        f"{dtype} | {mode} | {n_layers} layers  "
        f"(dots = T_actual per layer on right axis)",
        fontsize=13,
    )
    ax_left.grid(True, which="both", linestyle="--", alpha=0.35)

    # Legend 1: HW (colour + marker) + line style key
    hw_handles = [
        mlines.Line2D([], [], color=cmap(i % 10),
                      marker=_HW_MARKERS[i % len(_HW_MARKERS)],
                      markersize=7, linestyle="None", label=r.hw.name)
        for i, r in enumerate(results_list)
    ]
    hw_handles += [
        mlines.Line2D([], [], color="black", linewidth=1.8, linestyle="-",
                      label="solid = Perf ceiling (left)"),
        mlines.Line2D([], [], color="gray", linewidth=0, marker="o",
                      markersize=7, label="dot = T_actual (right)"),
    ]
    hw_leg = ax_left.legend(handles=hw_handles, loc="upper left",
                             fontsize=9, framealpha=0.88, title="Hardware")
    ax_left.add_artist(hw_leg)

    # Legend 2: layer-type colours
    all_types = sorted({lm["layer_type"] for m in metrics_list
                        for lm in m["layers"]
                        if lm["layer_type"] not in ("Unknown", "Dropout", "Reshape")})
    type_handles = [
        mpatches.Patch(color=_TYPE_COLORS.get(lt, _DEFAULT_COLOR), label=lt)
        for lt in all_types
    ]
    if type_handles:
        ax_left.legend(handles=type_handles, loc="lower left", fontsize=7,
                       framealpha=0.88, title="Layer type (dot fill colour)",
                       title_fontsize=7,
                       ncol=max(1, len(type_handles) // 6 + 1))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Set1] Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Plot Set 2 — All models × all HW, model-aggregate timing
# ---------------------------------------------------------------------------

def plot_set2_aggregate_time(
    results_grid: dict,
    metrics_grid: dict,
    hw_targets: list,
    save_path: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """Single dual-axis plot: 4 HW rooflines + 16 model-aggregate T_actual dots."""
    first_results = next(iter(results_grid.values()))
    dtype         = first_results[0].dtype
    mode          = first_results[0].mode
    model_names   = list(results_grid.keys())
    cmap          = plt.get_cmap("tab10")

    # Gather aggregate metrics per (model, HW)
    agg: dict = {}
    all_ai_vals: list = []
    for model_name, metrics_list in metrics_grid.items():
        agg[model_name] = []
        for m in metrics_list:
            ai    = m["model"]["I_model"]
            T_act = m["model"]["T_actual_model"]
            W_tot = m["model"]["W_total"]
            agg[model_name].append((max(ai, 1e-3), T_act, W_tot))
            all_ai_vals.append(max(ai, 1e-3))

    all_ridges = [hw.ridge_point(dtype) for hw in hw_targets]
    ai_min = min(min(all_ai_vals) * 0.3, 1e-2) if all_ai_vals else 1e-2
    ai_max = max(max(all_ridges) * 10, max(all_ai_vals) * 5, 1e4) if all_ai_vals else 1e4
    x_curve = np.logspace(np.log10(ai_min), np.log10(ai_max), 500)

    fig, ax_left = plt.subplots(figsize=(15, 9))
    ax_right = ax_left.twinx()

    # Left axis: HW roofline ceilings (solid)
    for hw_idx, hw in enumerate(hw_targets):
        color      = cmap(hw_idx % 10)
        peak_flops = hw._get_peak_flops(dtype)
        peak_bw    = hw.peak_mem_bw
        y_perf     = np.minimum(x_curve * peak_bw, peak_flops)
        ax_left.plot(x_curve, y_perf, color=color, linewidth=2.2,
                     linestyle="-", zorder=5)

    # Dot sizes proportional to log(total_flops) for visual model-size cue
    ref_flops  = [agg[m][0][2] for m in model_names]
    max_log_f  = max(np.log1p(f) for f in ref_flops) if ref_flops else 1.0

    annotated: set = set()
    for m_idx, model_name in enumerate(model_names):
        m_marker = _MODEL_MARKERS[m_idx % len(_MODEL_MARKERS)]
        W_ref    = agg[model_name][0][2]
        dot_size = max(120, 600 * (np.log1p(W_ref) / max_log_f) ** 1.5)
        best_ai_idx = int(np.argmax([a[0] for a in agg[model_name]]))

        for hw_idx, (ai, T_act, _) in enumerate(agg[model_name]):
            color = cmap(hw_idx % 10)
            ax_right.scatter([ai], [T_act],
                             marker=m_marker, c=[color],
                             s=dot_size, edgecolors="white", linewidths=0.9,
                             zorder=13, alpha=0.92)

        if model_name not in annotated:
            annotated.add(model_name)
            best_ai, best_T, _ = agg[model_name][best_ai_idx]
            ax_right.annotate(
                model_name,
                xy=(best_ai, best_T),
                xytext=(best_ai * 1.2, best_T * 1.5),
                fontsize=9, fontweight="bold", color="#222222",
                arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.9),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#dddddd", alpha=0.9),
                zorder=22,
            )

    ax_left.set_xscale("log")
    ax_left.set_yscale("log")
    ax_right.set_yscale("log")

    ax_left.set_xlabel("Arithmetic Intensity (FLOPs/Byte)", fontsize=12)
    ax_left.set_ylabel("Attainable Performance (FLOPs/s)", fontsize=12)
    ax_right.set_ylabel(
        "T_actual — Model Inference Time  (seconds, log)", fontsize=12)

    ax_left.yaxis.set_major_formatter(plt.FuncFormatter(_perf_fmt_axis))
    ax_right.yaxis.set_major_formatter(plt.FuncFormatter(_time_fmt_axis))

    ax_left.set_title(
        f"Roofline + Model Inference Time  —  All Models × All Hardware\n"
        f"{dtype} | {mode}  "
        f"(dots = T_actual per model on right axis;  size ~ log FLOPs)",
        fontsize=13,
    )
    ax_left.grid(True, which="both", linestyle="--", alpha=0.35)

    # Legend: HW (line colour)
    hw_handles = [
        mlines.Line2D([], [], color=cmap(i % 10), linewidth=2.2, linestyle="-",
                      label=hw.name)
        for i, hw in enumerate(hw_targets)
    ]
    hw_handles.append(
        mlines.Line2D([], [], color="black", linewidth=1.8, linestyle="-",
                      label="solid = Perf ceiling (left)")
    )
    hw_leg = ax_left.legend(handles=hw_handles, loc="upper left",
                             fontsize=9, framealpha=0.88, title="Hardware (line colour)")
    ax_left.add_artist(hw_leg)

    # Legend: model (marker shape)
    model_handles = [
        mlines.Line2D([], [], color="gray",
                      marker=_MODEL_MARKERS[i % len(_MODEL_MARKERS)],
                      markersize=8, linestyle="None", label=mn)
        for i, mn in enumerate(model_names)
    ]
    model_handles.append(
        mlines.Line2D([], [], color="gray", marker="o", markersize=7,
                      linestyle="None", label="dot = T_actual (right)")
    )
    ax_right.legend(handles=model_handles, loc="lower right",
                    fontsize=9, framealpha=0.88, title="Model (marker shape)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Set2] Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Plot Set 3 — Per-model per-HW layerwise T_memory / T_compute bars
# ---------------------------------------------------------------------------

def plot_set3_layerwise_bars(
    model_name: str,
    metrics: dict,
    result,
    save_path: str | None = None,
    show: bool = False,
) -> plt.Figure | None:
    """Dual-axis roofline + adjacent T_memory / T_compute bars for one model + HW.

    Left Y:  Attainable performance  [roofline curve + layer-position dots].
    Right Y: T_memory (blue) and T_compute (orange) adjacent bar pairs.
    Below X: truncated layer name for each bar group.
    """
    hw         = result.hw
    dtype      = result.dtype
    peak_bw    = hw.peak_mem_bw
    peak_flops = hw._get_peak_flops(dtype)

    selected = select_top5_per_type(metrics["layers"])
    if not selected:
        print(f"[Set3] No valid layers for {model_name} on {hw.name}; skipping.")
        return None

    # Roofline x-range
    all_ai = [max(lm["I"], 1e-3) for lm in selected]
    ridge  = hw.ridge_point(dtype)
    ai_min = min(min(all_ai) * 0.25, 1e-2)
    ai_max = max(ridge * 12, max(all_ai) * 6, 1e3)
    x_curve = np.logspace(np.log10(ai_min), np.log10(ai_max), 500)

    fig, ax_left = plt.subplots(figsize=(max(16, len(selected) * 0.9), 9))
    fig.subplots_adjust(bottom=0.30)   # room for vertical layer name labels
    ax_right = ax_left.twinx()

    # Left axis: single HW roofline (solid)
    hw_color   = "#2D6A9F"
    y_perf     = np.minimum(x_curve * peak_bw, peak_flops)
    ax_left.plot(x_curve, y_perf, color=hw_color, linewidth=2.5,
                 linestyle="-", zorder=5, label=f"{hw.name} roofline")

    # Left axis: reference dots at (I_layer, Y_layer) coloured by layer type
    ref_dot_colors = [_TYPE_COLORS.get(lm["layer_type"], _DEFAULT_COLOR)
                      for lm in selected]
    ax_left.scatter(all_ai, [lm["Y"] for lm in selected],
                    c=ref_dot_colors, s=65, alpha=0.80,
                    edgecolors=hw_color, linewidths=0.9, zorder=9)

    # Right axis: adjacent bar pairs
    # Each bar pair is centred at x_c = lm["I"].
    # Left  bar (T_memory):  [x_c * (1 - 2*f), x_c * (1 - f)]
    # Right bar (T_compute): [x_c * (1 - f),   x_c           ]
    # where f = 0.10  → 10 % of the layer's AI value per bar.
    BAR_FRAC = 0.10

    label_x_positions: list = []
    label_names:       list = []
    type_label_colors: list = []

    for lm in selected:
        x_c   = max(lm["I"], 1e-3)
        delta = x_c * BAR_FRAC

        # T_memory bar (left)
        x_mem = x_c * (1.0 - 2.0 * BAR_FRAC)
        ax_right.bar(x_mem, max(lm["T_memory"], 1e-20),
                     width=delta, align="edge",
                     color=_BAR_COLOR_MEM, alpha=0.78,
                     edgecolor="white", linewidth=0.4, zorder=10)

        # T_compute bar (right)
        x_comp = x_c * (1.0 - BAR_FRAC)
        ax_right.bar(x_comp, max(lm["T_compute"], 1e-20),
                     width=delta, align="edge",
                     color=_BAR_COLOR_COMP, alpha=0.78,
                     edgecolor="white", linewidth=0.4, zorder=10)

        # T_actual indicator line across both bars
        T_act = lm["T_actual"]
        if T_act > 0:
            ax_right.hlines(T_act,
                            xmin=x_mem,
                            xmax=x_comp + delta,
                            colors="#333333", linewidths=1.1,
                            linestyles="--", zorder=11, alpha=0.7)

        label_x_positions.append(x_c)
        # Compact layer name: use last two dotted segments if available
        parts = lm["name"].split(".")
        if len(parts) >= 2:
            short = ".".join(parts[-2:])
        else:
            short = lm["name"]
        short = short[:25]
        label_names.append(short)
        type_label_colors.append(
            _TYPE_COLORS.get(lm["layer_type"], _DEFAULT_COLOR))

    # Layer name labels placed below the X axis using blended coordinates
    # get_xaxis_transform(): x in data coords, y in axes fraction (0 = bottom)
    xaxis_transform = ax_left.get_xaxis_transform()
    for x_pos, name in zip(label_x_positions, label_names):
        ax_left.text(x_pos, -0.04, name,
                     transform=xaxis_transform,
                     ha="center", va="top",
                     fontsize=7.5, rotation=90, color="#333333",
                     clip_on=False)

    # Divider line just below X axis to visually separate layer names
    ax_left.axhline(y=ax_left.get_ylim()[0], color="#aaaaaa",
                    linewidth=0.5, linestyle="-")

    # Axis formatting
    ax_left.set_xscale("log")
    ax_left.set_yscale("log")
    ax_right.set_yscale("log")

    ax_left.set_xlabel("Arithmetic Intensity (FLOPs/Byte)", fontsize=12)
    ax_left.set_ylabel("Attainable Performance (FLOPs/s)", fontsize=12)
    ax_right.set_ylabel(
        "Time (seconds)  [blue = T_memory  |  orange = T_compute]",
        fontsize=12)

    ax_left.yaxis.set_major_formatter(plt.FuncFormatter(_perf_fmt_axis))
    ax_right.yaxis.set_major_formatter(plt.FuncFormatter(_time_fmt_axis))

    ax_left.set_title(
        f"Layerwise Time Analysis — {model_name}  on  {hw.name}\n"
        f"{dtype}  |  top-5 instances per layer type  |  "
        f"dashed line = T_actual = max(T_compute, T_memory)",
        fontsize=12,
    )
    ax_left.grid(True, which="major", linestyle="--", alpha=0.30)
    ax_left.grid(True, which="minor", linestyle=":", alpha=0.15)

    # Legend (upper left)
    roofline_h = mlines.Line2D([], [], color=hw_color, linewidth=2.2,
                                linestyle="-", label=f"Roofline: {hw.name} (left axis)")
    dot_h      = mlines.Line2D([], [], color=hw_color, marker="o",
                                linestyle="None", markersize=6,
                                label="Attainable perf dot (left axis)")
    mem_h      = mpatches.Patch(color=_BAR_COLOR_MEM,  alpha=0.78,
                                label="T_memory (right axis)")
    comp_h     = mpatches.Patch(color=_BAR_COLOR_COMP, alpha=0.78,
                                label="T_compute (right axis)")
    tact_h     = mlines.Line2D([], [], color="#333333", linewidth=1.1,
                                linestyle="--", label="T_actual = max(T_c, T_m)")
    ax_left.legend(handles=[roofline_h, dot_h, mem_h, comp_h, tact_h],
                   loc="upper left", fontsize=9, framealpha=0.90)

    # Layer-type colour legend (upper right)
    present_types = sorted({lm["layer_type"] for lm in selected})
    type_handles  = [
        mpatches.Patch(color=_TYPE_COLORS.get(lt, _DEFAULT_COLOR), label=lt)
        for lt in present_types
    ]
    if type_handles:
        ax_right.legend(handles=type_handles, loc="upper right", fontsize=7,
                        framealpha=0.88, title="Layer type  (dot colour)",
                        title_fontsize=7)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Set3] Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Report generation — TXT / CSV / Markdown / HTML
# ---------------------------------------------------------------------------

# ---- Data builders ---------------------------------------------------------

def _model_summary_rows(results_grid: dict, metrics_grid: dict) -> list:
    """One row per (model × HW) with model-level timing metrics."""
    rows = []
    for model_name, metrics_list in metrics_grid.items():
        for hw_idx, m in enumerate(metrics_list):
            agg = m["model"]
            hw  = m["hw"]
            rows.append({
                "Model":      model_name,
                "Hardware":   hw.name,
                "W (FLOPs)":  _fmt_flops(agg["W_total"]),
                "Q (Bytes)":  _fmt_flops(agg["Q_total"]),
                "I (F/B)":    f"{agg['I_model']:.3f}",
                "Y (FLOPs/s)":_fmt_flops(agg["Y_model"]),
                "T_compute":  _fmt_time(agg["T_compute_model"]),
                "T_memory":   _fmt_time(agg["T_memory_model"]),
                "T_actual":   _fmt_time(agg["T_actual_model"]),
                "Bottleneck": "compute" if agg["T_compute_model"] >= agg["T_memory_model"] else "memory",
                # raw seconds for sorting / colour coding
                "_T_actual_s": agg["T_actual_model"],
            })
    return rows


def _layer_rows(results_grid: dict, metrics_grid: dict) -> list:
    """One row per (model × HW × layer) for all layers."""
    rows = []
    for model_name, metrics_list in metrics_grid.items():
        for m in metrics_list:
            hw = m["hw"]
            for lm in m["layers"]:
                rows.append({
                    "Model":      model_name,
                    "Hardware":   hw.name,
                    "Layer":      lm["name"],
                    "Type":       lm["layer_type"],
                    "W (FLOPs)":  _fmt_flops(lm["W"]),
                    "Q (Bytes)":  _fmt_flops(lm["Q"]),
                    "I (F/B)":    f"{lm['I']:.4f}",
                    "Y (FLOPs/s)":_fmt_flops(lm["Y"]),
                    "T_compute":  _fmt_time(lm["T_compute"]),
                    "T_memory":   _fmt_time(lm["T_memory"]),
                    "T_actual":   _fmt_time(lm["T_actual"]),
                    "Bottleneck": lm["bottleneck"],
                    "_T_actual_s": lm["T_actual"],
                })
    return rows


def _top10_rows(metrics_grid: dict) -> list:
    """Top-10 unique layer types per model (by W), one row per (model × layer_type)
    with T_actual columns for each HW side-by-side."""
    rows = []
    for model_name, metrics_list in metrics_grid.items():
        top10 = select_top10_unique_layers(metrics_list[0]["layers"])
        top10_names = {lm["name"] for lm in top10}
        for rank, ref_lm in enumerate(top10, start=1):
            row = {
                "Rank":       rank,
                "Model":      model_name,
                "Type":       ref_lm["layer_type"],
                "Layer":      ref_lm["name"],
                "W (FLOPs)":  _fmt_flops(ref_lm["W"]),
                "I (F/B)":    f"{ref_lm['I']:.4f}",
            }
            # Add one T_actual column per HW
            for hw_idx, m in enumerate(metrics_list):
                hw_name = m["hw"].name
                match = next((lm for lm in m["layers"] if lm["name"] == ref_lm["name"]), None)
                row[f"T_actual [{hw_name}]"] = _fmt_time(match["T_actual"]) if match else "N/A"
            rows.append(row)
    return rows


# ---- Format helpers --------------------------------------------------------

def _ascii_table(rows: list, title: str) -> str:
    """Render a list-of-dicts as a fixed-width ASCII table string."""
    if not rows:
        return f"  {title}: (no data)\n"
    # Strip internal _xxx keys
    visible_keys = [k for k in rows[0].keys() if not k.startswith("_")]
    col_widths = {k: max(len(k), max(len(str(r.get(k, ""))) for r in rows))
                  for k in visible_keys}
    sep   = "+-" + "-+-".join("-" * col_widths[k] for k in visible_keys) + "-+"
    hdr   = "| " + " | ".join(k.ljust(col_widths[k]) for k in visible_keys) + " |"
    lines = [f"\n  {title}", sep, hdr, sep]
    for r in rows:
        line = "| " + " | ".join(str(r.get(k, "")).ljust(col_widths[k]) for k in visible_keys) + " |"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines) + "\n"


def _md_table(rows: list, title: str) -> str:
    """Render a list-of-dicts as a Markdown table string."""
    if not rows:
        return f"### {title}\n\n_(no data)_\n"
    visible_keys = [k for k in rows[0].keys() if not k.startswith("_")]
    header = "| " + " | ".join(visible_keys) + " |"
    sep    = "| " + " | ".join("---" for _ in visible_keys) + " |"
    lines  = [f"### {title}", "", header, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in visible_keys) + " |")
    lines.append("")
    return "\n".join(lines)


def _csv_text(rows: list) -> str:
    """Render a list-of-dicts as CSV text."""
    import csv, io
    if not rows:
        return ""
    visible_keys = [k for k in rows[0].keys() if not k.startswith("_")]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=visible_keys, extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _html_table(rows: list, title: str) -> str:
    """Render a list-of-dicts as an HTML table with heat-map colouring on T_actual."""
    if not rows:
        return f"<h3>{title}</h3><p><em>No data.</em></p>"
    visible_keys = [k for k in rows[0].keys() if not k.startswith("_")]

    # Heat-map colouring: scale T_actual values to green-yellow-red
    t_vals = [r.get("_T_actual_s", 0) for r in rows]
    t_min  = min(t_vals) if t_vals else 0
    t_max  = max(t_vals) if t_vals else 1

    def _cell_bg(t: float) -> str:
        if t_max <= t_min:
            return "#ffffff"
        ratio = (t - t_min) / (t_max - t_min)   # 0 = fast (green) → 1 = slow (red)
        r = int(60  + 185 * ratio)
        g = int(200 - 150 * ratio)
        b = int(80  -  60 * ratio)
        return f"rgb({r},{g},{b})"

    T_actual_key = "T_actual"
    html_parts = [
        f"<h3>{title}</h3>",
        '<table border="1" cellpadding="5" cellspacing="0" '
        'style="border-collapse:collapse;font-family:monospace;font-size:13px;">',
        "<thead><tr>",
    ]
    for k in visible_keys:
        html_parts.append(
            f'<th style="background:#2c3e50;color:#ecf0f1;padding:6px 10px;">{k}</th>')
    html_parts.append("</tr></thead><tbody>")

    for i, r in enumerate(rows):
        bg_row = "#f9f9f9" if i % 2 == 0 else "#ffffff"
        html_parts.append(f'<tr style="background:{bg_row};">')
        for k in visible_keys:
            val = str(r.get(k, ""))
            if k == T_actual_key:
                cell_bg = _cell_bg(r.get("_T_actual_s", 0))
                html_parts.append(
                    f'<td style="background:{cell_bg};font-weight:bold;'
                    f'padding:5px 10px;">{val}</td>')
            elif k == "Bottleneck":
                color = "#c0392b" if val == "compute" else "#1a5276"
                html_parts.append(
                    f'<td style="color:{color};font-weight:bold;padding:5px 10px;">{val}</td>')
            else:
                html_parts.append(f'<td style="padding:5px 10px;">{val}</td>')
        html_parts.append("</tr>")

    html_parts.append("</tbody></table>")
    return "\n".join(html_parts)


# ---- Report writers --------------------------------------------------------

def save_reports(results_grid: dict, metrics_grid: dict) -> None:
    """Generate timing reports in TXT, CSV, Markdown, and HTML formats."""
    model_rows = _model_summary_rows(results_grid, metrics_grid)
    layer_rows = _layer_rows(results_grid, metrics_grid)
    top10_rows = _top10_rows(metrics_grid)

    tables = [
        ("Model Summary",       model_rows),
        ("Top-10 Layer Summary", top10_rows),
        ("Full Layer Detail",    layer_rows),
    ]

    # ---- TXT ---------------------------------------------------------------
    txt_path = "time_analysis_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Time Analysis Roofline Report\n")
        f.write(f"Models : {', '.join(MODELS)}\n")
        f.write(f"HW     : {', '.join(hw.name for hw in HW_TARGETS)}\n")
        f.write(f"dtype  : {DTYPE}  |  mode : {MODE}\n")
        f.write("=" * 120 + "\n")
        for title, rows in tables:
            f.write(_ascii_table(rows, title))
            f.write("\n")
    print(f"[Report] Saved: {txt_path}")

    # ---- CSV ---------------------------------------------------------------
    csv_names = {
        "Model Summary":        "time_report_model_summary.csv",
        "Top-10 Layer Summary": "time_report_top10_layers.csv",
        "Full Layer Detail":    "time_report_layer_detail.csv",
    }
    for title, rows in tables:
        csv_path = csv_names[title]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            f.write(_csv_text(rows))
        print(f"[Report] Saved: {csv_path}")

    # ---- Markdown ----------------------------------------------------------
    md_path = "time_analysis_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Time Analysis Roofline Report\n\n")
        f.write(f"- **Models**: {', '.join(MODELS)}\n")
        f.write(f"- **Hardware**: {', '.join(hw.name for hw in HW_TARGETS)}\n")
        f.write(f"- **dtype**: {DTYPE}  |  **mode**: {MODE}\n\n")
        for title, rows in tables:
            f.write(_md_table(rows, title))
            f.write("\n")
    print(f"[Report] Saved: {md_path}")

    # ---- HTML --------------------------------------------------------------
    html_path = "time_analysis_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<!DOCTYPE html><html><head>"
                '<meta charset="utf-8">'
                "<title>Time Analysis Roofline Report</title>"
                "<style>body{font-family:Arial,sans-serif;margin:30px;}"
                "h1{color:#2c3e50;}h3{color:#34495e;margin-top:30px;}</style>"
                "</head><body>")
        f.write("<h1>Time Analysis Roofline Report</h1>")
        f.write(f"<p><b>Models:</b> {', '.join(MODELS)}<br>")
        f.write(f"<b>Hardware:</b> {', '.join(hw.name for hw in HW_TARGETS)}<br>")
        f.write(f"<b>dtype:</b> {DTYPE} &nbsp;|&nbsp; <b>mode:</b> {MODE}</p>")
        for title, rows in tables:
            f.write(_html_table(rows, title))
            f.write("<br>")
        f.write("</body></html>")
    print(f"[Report] Saved: {html_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _hw_slug(hw) -> str:
    """Filesystem-safe slug from an HWSpec name."""
    return hw.name.lower().replace(" ", "_").replace("/", "").replace("(", "").replace(")", "")


def _model_slug(model_name: str) -> str:
    return model_name.lower().replace(" ", "_")


def main() -> None:
    sep = "=" * 100
    print(sep)
    print("  Time Analysis Roofline Demo")
    print(f"  Models : {', '.join(MODELS)}")
    print(f"  HW     : {', '.join(hw.name for hw in HW_TARGETS)}")
    print(f"  dtype={DTYPE}  mode={MODE}")
    print(sep)

    # ---- Build results and metrics grids -----------------------------------
    results_grid: dict = {}    # {model_name: [AnalysisResults, ...] per HW}
    metrics_grid: dict = {}    # {model_name: [metrics_dict, ...] per HW}

    for model_name, (model_fn, input_shapes) in MODELS.items():
        thin = "-" * 80
        print(f"\n{thin}")
        print(f"  Analysing: {model_name}")
        print(thin)
        model_obj    = model_fn(weights=None)
        results_list = []
        metrics_list = []

        for hw in HW_TARGETS:
            result  = analyze(model=model_obj, input_shapes=input_shapes,
                              hw=hw, dtype=DTYPE, mode=MODE)
            metrics = compute_time_metrics(result)
            results_list.append(result)
            metrics_list.append(metrics)

            m = metrics["model"]
            print(
                f"    {hw.name:<28}  "
                f"W={_fmt_flops(m['W_total'])} FLOPs  "
                f"Q={_fmt_flops(m['Q_total'])} B  "
                f"I={m['I_model']:.2f} F/B  "
                f"T_compute={_fmt_time(m['T_compute_model'])}  "
                f"T_memory={_fmt_time(m['T_memory_model'])}  "
                f"-> T_actual={_fmt_time(m['T_actual_model'])}"
            )

        results_grid[model_name] = results_list
        metrics_grid[model_name] = metrics_list

    # ---- Set 1: Per-model plots (4 figures) --------------------------------
    print(f"\n{sep}")
    print("  Generating Set 1 — per-model time plots (4 figures) …")
    print(sep)
    for model_name in MODELS:
        out = f"time_set1_{_model_slug(model_name)}.png"
        plot_set1_model_time(
            model_name   = model_name,
            results_list = results_grid[model_name],
            metrics_list = metrics_grid[model_name],
            save_path    = out,
            show         = False,
        )

    # ---- Set 2: Aggregate plot (1 figure) ----------------------------------
    print(f"\n{sep}")
    print("  Generating Set 2 — aggregate model × HW timing (1 figure) …")
    print(sep)
    plot_set2_aggregate_time(
        results_grid = results_grid,
        metrics_grid = metrics_grid,
        hw_targets   = HW_TARGETS,
        save_path    = "time_set2_aggregate.png",
        show         = False,
    )

    # ---- Set 3: Layerwise bars per HW per model (16 figures) ---------------
    print(f"\n{sep}")
    print("  Generating Set 3 — layerwise bar plots (16 figures) …")
    print(sep)
    for hw_idx, hw in enumerate(HW_TARGETS):
        hs = _hw_slug(hw)
        for model_name in MODELS:
            ms  = _model_slug(model_name)
            out = f"time_set3_{hs}_{ms}.png"
            plot_set3_layerwise_bars(
                model_name = model_name,
                metrics    = metrics_grid[model_name][hw_idx],
                result     = results_grid[model_name][hw_idx],
                save_path  = out,
                show       = False,
            )

    # ---- Reports (TXT / CSV / Markdown / HTML) -----------------------------
    print(f"\n{sep}")
    print("  Generating reports (TXT / CSV / Markdown / HTML) ...")
    print(sep)
    save_reports(results_grid, metrics_grid)

    # ---- Summary -----------------------------------------------------------
    print(f"\n{sep}")
    print("  Done.  Output files written:")
    for model_name in MODELS:
        ms = _model_slug(model_name)
        print(f"    time_set1_{ms}.png")
    print("    time_set2_aggregate.png")
    for hw in HW_TARGETS:
        hs = _hw_slug(hw)
        for model_name in MODELS:
            ms = _model_slug(model_name)
            print(f"    time_set3_{hs}_{ms}.png")
    print("  Reports:")
    print("    time_analysis_report.txt")
    print("    time_analysis_report.md")
    print("    time_analysis_report.html")
    print("    time_report_model_summary.csv")
    print("    time_report_top10_layers.csv")
    print("    time_report_layer_detail.csv")
    print(sep)


if __name__ == "__main__":
    main()
