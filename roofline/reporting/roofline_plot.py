"""RooflinePlot — matplotlib roofline chart with per-layer scatter overlay."""

from __future__ import annotations

from typing import Optional
import warnings

import numpy as np


# Layer type → colour (consistent across all plots)
_TYPE_COLORS = {
    "Linear": "#4C72B0",
    "MatMul": "#4C72B0",
    "Gemm": "#4C72B0",
    "Conv1d": "#DD8452",
    "Conv2d": "#DD8452",
    "Conv3d": "#DD8452",
    "ConvTranspose1d": "#DD8452",
    "ConvTranspose2d": "#DD8452",
    "ConvTranspose3d": "#DD8452",
    "MultiHeadAttention": "#55A868",
    "Attention": "#55A868",
    "LSTM": "#C44E52",
    "GRU": "#C44E52",
    "RNN": "#C44E52",
    "BatchNorm": "#8172B3",
    "LayerNorm": "#8172B3",
    "GroupNorm": "#8172B3",
    "InstanceNorm": "#8172B3",
    "Embedding": "#937860",
    "ReLU": "#DA8BC3",
    "GELU": "#DA8BC3",
    "SiLU": "#DA8BC3",
    "Activation": "#DA8BC3",
    "MaxPool": "#3A86FF",
    "AvgPool": "#06D6A0",
    "AdaptiveAvgPool": "#06D6A0",
    "Elementwise": "#CCB974",
    "Reshape": "#64B5CD",
    "Dropout": "#AAAAAA",
}
_DEFAULT_COLOR = "#888888"


def _fmt_flops_plain(n: int) -> str:
    if n >= 1e12: return f"{n/1e12:.2f}T"
    if n >= 1e9:  return f"{n/1e9:.2f}G"
    if n >= 1e6:  return f"{n/1e6:.1f}M"
    if n >= 1e3:  return f"{n/1e3:.0f}K"
    return str(n)


class RooflinePlot:
    """Generates the roofline chart for an ``AnalysisResults`` object."""

    def __init__(self, results):
        self.results = results

    def plot(
        self,
        figsize=(13, 7),
        save_path: Optional[str] = None,
        show: bool = True,
        annotate_top_n: int = 5,
        log_scale: bool = True,
        model_name: Optional[str] = None,
    ):
        """Draw the roofline plot.

        Parameters
        ----------
        figsize:
            Matplotlib figure size.
        save_path:
            If given, save the figure to this path (.png, .pdf, etc.).
        show:
            Call ``plt.show()`` after rendering.
        annotate_top_n:
            Annotate the N layers with the highest FLOPs.
        log_scale:
            Use log-log axes (strongly recommended for roofline plots).
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError as e:
            raise ImportError("matplotlib is required: pip install matplotlib") from e

        hw = self.results.hw
        dtype = self.results.dtype
        layers = self.results.layers

        peak_flops = hw._get_peak_flops(dtype)
        peak_bw = hw.peak_mem_bw
        ridge = hw.ridge_point(dtype)

        # ---- Build roofline curve ----
        ai_min = 1e-2
        ai_max = max(ridge * 10, 1e4)
        x_curve = np.logspace(np.log10(ai_min), np.log10(ai_max), 500)
        y_curve = np.minimum(x_curve * peak_bw, peak_flops)

        fig, ax = plt.subplots(figsize=figsize)

        # Roofline curve
        ax.plot(x_curve, y_curve, color="black", linewidth=2.5, label="Roofline", zorder=5)

        # Ridge point vertical dashed line
        ax.axvline(
            x=ridge, color="black", linestyle="--", linewidth=1.2, alpha=0.6,
            label=f"Ridge point = {ridge:.1f} FLOPs/B",
        )

        # Annotated region labels
        ax.text(
            ridge * 0.3, peak_flops * 0.55,
            "Memory\nBound", ha="center", va="center",
            fontsize=10, color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
        )
        ax.text(
            ridge * 3, peak_flops * 0.75,
            "Compute\nBound", ha="center", va="center",
            fontsize=10, color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.7),
        )

        # ---- Scatter layers ----
        # Separate meaningful layers from Unknown bias-only rows
        known_layers = [l for l in layers if l.layer.layer_type != "Unknown" and l.flops > 0]
        unknown_layers = [l for l in layers if l not in known_layers]

        ai_vals = [max(l.arithmetic_intensity, 1e-3) for l in known_layers]
        perf_vals = [max(l.attainable_perf, 1.0) for l in known_layers]
        flop_vals = [l.flops for l in known_layers]
        colors = [_TYPE_COLORS.get(l.layer.layer_type, _DEFAULT_COLOR) for l in known_layers]

        # Unknown (bias) layers — subtle grey mini-dots, no labels
        if unknown_layers:
            unk_ai = [max(l.arithmetic_intensity, 1e-3) for l in unknown_layers]
            unk_perf = [max(l.attainable_perf, 1.0) for l in unknown_layers]
            ax.scatter(unk_ai, unk_perf, c="#cccccc", s=15, alpha=0.4,
                       edgecolors="none", zorder=6)

        if not known_layers:
            # Fall back to plotting everything if nothing is "known"
            known_layers = layers
            ai_vals = [max(l.arithmetic_intensity, 1e-3) for l in known_layers]
            perf_vals = [max(l.attainable_perf, 1.0) for l in known_layers]
            flop_vals = [l.flops for l in known_layers]
            colors = [_TYPE_COLORS.get(l.layer.layer_type, _DEFAULT_COLOR) for l in known_layers]

        max_flops = max(flop_vals) if flop_vals else 1
        n = len(known_layers)
        sizes = [
            max(60, 600 * (np.log1p(f) / np.log1p(max_flops)) ** 2)
            for f in flop_vals
        ]

        # Apply a spread so stacked dots at the same AI are visually separated.
        # Sort by flops to assign spread positions consistently.
        sort_order = sorted(range(n), key=lambda i: flop_vals[i], reverse=True)
        spread = np.linspace(-0.6, 0.6, n)            # vertical spread in log decades
        h_nudge = np.linspace(-0.15, 0.15, n)         # tiny horizontal nudge
        perf_jittered = [None] * n
        ai_jittered = [None] * n
        for rank, i in enumerate(sort_order):
            perf_jittered[i] = perf_vals[rank] * (10 ** spread[rank])
            ai_jittered[i] = ai_vals[i] * (10 ** h_nudge[rank])

        sc = ax.scatter(
            ai_jittered, perf_jittered,
            c=colors, s=sizes, alpha=0.85, edgecolors="white",
            linewidths=1.0, zorder=10,
        )

        # ---- Number labels on dots + legend table ----
        # Sort by FLOPs descending for numbering; restrict to annotate_top_n
        sort_idx = sorted(range(n), key=lambda i: flop_vals[i], reverse=True)
        top_idx = sort_idx[:annotate_top_n] if annotate_top_n > 0 else []

        legend_lines = []
        for rank, i in enumerate(top_idx):
            num = rank + 1
            ai = ai_jittered[i]
            perf = perf_jittered[i]
            # Draw a white-backed number inside/beside the dot
            ax.text(
                ai, perf, str(num),
                ha="center", va="center",
                fontsize=7, fontweight="bold", color="white",
                zorder=20,
            )
            # Build clean short name for the legend table
            raw_name = known_layers[i].layer.name
            short = raw_name.split("/")[-1]   # last path segment
            short = short[:38] + "…" if len(short) > 38 else short
            layer_type = known_layers[i].layer.layer_type
            flops_str = _fmt_flops_plain(flop_vals[i])
            legend_lines.append(f"{num:>2}. [{layer_type}]  {short}  ({flops_str})")

        if legend_lines:
            legend_text = "Top layers by FLOPs:\n" + "\n".join(legend_lines)
            ax.text(
                0.99, 0.02, legend_text,
                transform=ax.transAxes,
                fontsize=7.5,
                verticalalignment="bottom",
                horizontalalignment="right",
                family="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="white",
                    edgecolor="#cccccc",
                    alpha=0.92,
                ),
                zorder=20,
            )

        # ---- Axes ----
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax.set_xlabel("Arithmetic Intensity  (FLOPs / Byte)", fontsize=12)
        ax.set_ylabel("Attainable Performance  (FLOPs / s)", fontsize=12)
        title_prefix = f"{model_name} — " if model_name else ""
        ax.set_title(
            f"{title_prefix}Roofline Model — {hw.name}\n"
            f"{dtype} | {self.results.mode} mode | {len(layers)} layers",
            fontsize=13,
        )

        # Y-axis: TFLOPS labels
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v/1e12:.0f}T" if v >= 1e12 else f"{v/1e9:.0f}G" if v >= 1e9 else f"{v:.0e}")
        )

        # ---- Legend ----
        seen_types = sorted({l.layer.layer_type for l in layers
                             if l.layer.layer_type != "Unknown"})
        handles = [
            mpatches.Patch(color=_TYPE_COLORS.get(t, _DEFAULT_COLOR), label=t)
            for t in seen_types
        ]
        handles += [
            plt.Line2D([0], [0], color="black", linewidth=2.5, label="Roofline"),
            plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.2,
                       label=f"Ridge ({ridge:.1f})"),
        ]
        ax.legend(
            handles=handles,
            loc="upper left",
            fontsize=8,
            framealpha=0.85,
            ncol=max(1, len(seen_types) // 8 + 1),
        )

        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[roofline] Plot saved to {save_path}")

        if show:
            plt.show()

        return fig, ax


def plot_model_across_hw(
    model_name: str,
    results_list,
    figsize=(14, 8),
    save_path: Optional[str] = None,
    show: bool = True,
    annotate_top_n: int = 8,
    log_scale: bool = True,
):
    """Draw one roofline figure for a single model across multiple hardware targets.

    All hardware roofline curves are overlaid on the same axes together with
    per-layer scatter points.  Because arithmetic intensity is hardware-independent,
    every layer lands at the same X position on all curves; only its Y (attainable
    performance) differs, making the cross-hardware comparison immediately visible.

    Parameters
    ----------
    model_name:
        Display name used in the plot title.
    results_list:
        List of ``AnalysisResults``, one per hardware target (same order as
        the HW_TARGETS list used to produce them).
    annotate_top_n:
        Annotate the N highest-FLOPs layers with their type label.
    """
    if not results_list:
        raise ValueError("results_list must contain at least one AnalysisResults object")

    try:
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches
    except ImportError as e:
        raise ImportError("matplotlib is required: pip install matplotlib") from e

    first_dtype = results_list[0].dtype
    first_mode = results_list[0].mode

    # Global AI range
    all_ai = [max(layer.arithmetic_intensity, 1e-3)
              for result in results_list for layer in result.layers]
    ai_min = min(min(all_ai) if all_ai else 1e-2, 1e-2)
    all_ridges = [r.hw.ridge_point(first_dtype) for r in results_list]
    ai_max = max(max(all_ridges) * 10, max(all_ai) * 10 if all_ai else 1e4, 1e4)

    x_curve = np.logspace(np.log10(ai_min), np.log10(ai_max), 500)
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("tab10")
    hw_markers = ["o", "s", "^", "D", "P", "X"]

    # Use FLOPs from first result (hardware-independent)
    ref_layers = [l for l in results_list[0].layers
                  if l.layer.layer_type != "Unknown" and l.flops > 0]
    top_layers_idx = sorted(range(len(ref_layers)),
                            key=lambda i: ref_layers[i].flops, reverse=True)[:annotate_top_n]

    for idx, result in enumerate(results_list):
        hw = result.hw
        hw_color = cmap(idx % 10)
        marker = hw_markers[idx % len(hw_markers)]
        peak_flops = hw._get_peak_flops(first_dtype)
        peak_bw = hw.peak_mem_bw
        ridge = hw.ridge_point(first_dtype)

        y_curve = np.minimum(x_curve * peak_bw, peak_flops)
        ax.plot(x_curve, y_curve, color=hw_color, linewidth=2.2,
                label=hw.name, zorder=5)
        ax.axvline(x=ridge, color=hw_color, linestyle="--", linewidth=1.0,
                   alpha=0.7, zorder=4)
        ax.scatter([ridge], [peak_flops], marker=marker, color=hw_color,
                   edgecolors="black", s=110, zorder=10)

        known = [l for l in result.layers
                 if l.layer.layer_type != "Unknown" and l.flops > 0]
        if known:
            ai_vals   = [max(l.arithmetic_intensity, 1e-3) for l in known]
            perf_vals = [max(l.attainable_perf, 1.0) for l in known]
            # Color each dot by its layer type so the "Layer types in AI" legend
            # is live.  Marker shape encodes which HW target the dot belongs to.
            dot_colors = [_TYPE_COLORS.get(l.layer.layer_type, _DEFAULT_COLOR)
                          for l in known]
            ax.scatter(ai_vals, perf_vals, c=dot_colors, marker=marker,
                       s=45, alpha=0.75, edgecolors=hw_color,
                       linewidths=1.0, zorder=11)

    # Annotate one representative per unique layer type (highest FLOPs for that type).
    # Using the highest-performing HW result for Y positions so labels sit at the
    # most visible (tallest) scatter points.
    best_result = max(results_list,
                      key=lambda r: r.hw._get_peak_flops(first_dtype))
    best_known = [l for l in best_result.layers
                  if l.layer.layer_type != "Unknown" and l.flops > 0]

    # Build {layer_type: best LayerStats by FLOPs}
    type_rep: dict = {}
    for ls in best_known:
        lt = ls.layer.layer_type
        if lt not in type_rep or ls.flops > type_rep[lt].flops:
            type_rep[lt] = ls

    # Sort by FLOPs descending so the most important types are prioritised
    # if the plot gets crowded; cap at annotate_top_n distinct types
    sorted_types = sorted(type_rep.values(), key=lambda l: l.flops, reverse=True)
    to_annotate = sorted_types[:annotate_top_n] if annotate_top_n > 0 else sorted_types

    # Spread text offsets to reduce overlap: alternate above/below the point
    offset_factors = [
        (1.7, 1.6), (1.7, 0.45), (0.35, 1.6), (0.35, 0.45),
        (1.7, 2.5), (0.15, 2.5), (2.5, 1.0), (0.15, 0.25),
    ]
    for rank, layer_stat in enumerate(to_annotate):
        ai = max(layer_stat.arithmetic_intensity, 1e-3)
        perf = max(layer_stat.attainable_perf, 1.0)
        ox, oy = offset_factors[rank % len(offset_factors)]
        ax.annotate(
            layer_stat.layer.layer_type,
            xy=(ai, perf),
            xytext=(ai * ox, perf * oy),
            fontsize=7.5,
            color="#333333",
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#cccccc", alpha=0.88),
            zorder=20,
        )

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel("Arithmetic Intensity  (FLOPs / Byte)", fontsize=12)
    ax.set_ylabel("Attainable Performance  (FLOPs / s)", fontsize=12)
    ax.set_title(
        f"{model_name} — Roofline across Hardware Targets\n"
        f"{first_dtype} | {first_mode} | {len(results_list[0].layers)} layers",
        fontsize=13,
    )
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v/1e12:.0f}T" if v >= 1e12
                          else f"{v/1e9:.0f}G" if v >= 1e9 else f"{v:.0e}")
    )

    # Legend 1 (upper-left): HW roofline curves — color = HW, shape = HW
    hw_handles = []
    for idx, result in enumerate(results_list):
        color = cmap(idx % 10)
        marker = hw_markers[idx % len(hw_markers)]
        hw_handles.append(
            mlines.Line2D([], [], color=color, marker=marker, markersize=7,
                          linewidth=2.0, label=result.hw.name)
        )
    hw_legend = ax.legend(handles=hw_handles, loc="upper left", fontsize=9,
                          framealpha=0.88,
                          title="Hardware  (curve color + dot border + shape)")
    ax.add_artist(hw_legend)   # keep it when adding the second legend

    # Legend 2 (lower-right): all layer types present in the scatter,
    # coloured with the global _TYPE_COLORS palette.
    # Dot fill color = layer type, dot border color = HW.
    all_layer_types = sorted({
        l.layer.layer_type
        for result in results_list
        for l in result.layers
        if l.layer.layer_type != "Unknown" and l.flops > 0
    })
    type_handles = [
        mpatches.Patch(
            color=_TYPE_COLORS.get(lt, _DEFAULT_COLOR),
            label=lt,
        )
        for lt in all_layer_types
    ]
    if type_handles:
        ax.legend(
            handles=type_handles,
            loc="lower right",
            fontsize=8,
            framealpha=0.88,
            title="Layer types  (dot fill color)",
            title_fontsize=8,
            ncol=max(1, len(type_handles) // 6 + 1),
        )
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[roofline] Saved: {save_path}")
    if show:
        plt.show()
    return fig


def plot_multiple(
    results_list,
    figsize=(13, 7),
    save_path: Optional[str] = None,
    show: bool = True,
    log_scale: bool = True,
):
    """Draw multiple roofline curves on the same axes for comparison."""
    if not results_list:
        raise ValueError("results_list must contain at least one AnalysisResults object")

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required: pip install matplotlib") from e

    first_dtype = results_list[0].dtype
    first_mode = results_list[0].mode
    for result in results_list:
        if result.dtype != first_dtype:
            raise ValueError("All results must use the same dtype for a combined roofline plot")
        if result.mode != first_mode:
            raise ValueError("All results must use the same mode (inference/training) for a combined plot")

    all_ai = [max(layer.arithmetic_intensity, 1e-3)
              for result in results_list for layer in result.layers]
    ai_min = min(all_ai) if all_ai else 1e-2
    ai_min = min(ai_min, 1e-2)
    all_ridges = [result.hw.ridge_point(first_dtype) for result in results_list]
    ai_max = max(max(all_ridges) * 10, max(all_ai) * 10 if all_ai else 1e4, 1e4)

    x_curve = np.logspace(np.log10(ai_min), np.log10(ai_max), 500)
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "P", "X"]

    for idx, result in enumerate(results_list):
        hw = result.hw
        color = cmap(idx % 10)
        peak_flops = hw._get_peak_flops(first_dtype)
        peak_bw = hw.peak_mem_bw
        ridge = hw.ridge_point(first_dtype)

        y_curve = np.minimum(x_curve * peak_bw, peak_flops)
        ax.plot(
            x_curve,
            y_curve,
            color=color,
            linewidth=2.3,
            label=f"{hw.name}",
            zorder=5,
        )
        ax.axvline(
            x=ridge,
            color=color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            zorder=4,
        )
        ax.scatter(
            [ridge],
            [peak_flops],
            marker=markers[idx % len(markers)],
            color=color,
            edgecolors="black",
            s=120,
            zorder=10,
            label=f"{hw.name} peak",
        )
        ax.text(
            ridge * 1.08,
            peak_flops * 0.92,
            f"{hw.name}",
            color=color,
            fontsize=9,
            weight="bold",
            zorder=12,
        )

        known_layers = [l for l in result.layers if l.layer.layer_type != "Unknown" and l.flops > 0]
        if known_layers:
            ai_vals = [max(l.arithmetic_intensity, 1e-3) for l in known_layers]
            perf_vals = [max(l.attainable_perf, 1.0) for l in known_layers]
            ax.scatter(
                ai_vals,
                perf_vals,
                c=color,
                marker=markers[idx % len(markers)],
                s=40,
                alpha=0.5,
                edgecolors="white",
                linewidths=0.7,
                zorder=11,
                label=f"{hw.name} layers",
            )

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel("Arithmetic Intensity  (FLOPs / Byte)", fontsize=12)
    ax.set_ylabel("Attainable Performance  (FLOPs / s)", fontsize=12)
    ax.set_title(
        f"Roofline Comparison — {first_dtype} | {first_mode}",
        fontsize=14,
    )
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v/1e12:.0f}T" if v >= 1e12 else f"{v/1e9:.0f}G" if v >= 1e9 else f"{v:.0e}")
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85, ncol=1)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    if show:
        plt.show()
    return fig


def plot_multi_model_multi_hw(
    results_grid: dict,
    figsize=(14, 9),
    save_path: Optional[str] = None,
    show: bool = True,
    log_scale: bool = True,
):
    """Draw a unified roofline comparison across multiple models and hardware targets.

    Uses **model-level aggregate metrics** (one dot per model-hardware pair) rather
    than per-layer scatter, keeping the combined view clean and readable.

    Each model forms a vertical cluster of same-shaped dots at its aggregate
    arithmetic intensity (AI = total_flops / total_bytes, hardware-independent).
    The 4 hardware-coloured dots in each cluster show how each hardware's roofline
    ceiling clips the model's attainable performance differently.

    Parameters
    ----------
    results_grid:
        ``{model_name: List[AnalysisResults]}`` — one list per model, each list
        containing one ``AnalysisResults`` per hardware target (same HW order for
        every model).
    """
    if not results_grid:
        raise ValueError("results_grid must not be empty")

    try:
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches
    except ImportError as e:
        raise ImportError("matplotlib is required: pip install matplotlib") from e

    # Collect metadata from first model's first result
    first_results_list = next(iter(results_grid.values()))
    first_dtype = first_results_list[0].dtype
    first_mode = first_results_list[0].mode
    hw_list = [r.hw for r in first_results_list]
    n_hw = len(hw_list)

    cmap = plt.get_cmap("tab10")
    hw_markers = ["o", "s", "^", "D", "P", "X"]
    # One distinct marker per model
    model_names = list(results_grid.keys())
    model_markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h"]

    # Compute aggregate AI and attainable perf per (model, hw)
    agg = {}   # {model_name: [(ai, attainable_perf, total_flops), ...]} indexed by HW
    all_ai_vals = []
    for model_name, results_list in results_grid.items():
        agg[model_name] = []
        for result in results_list:
            total_flops = sum(l.flops for l in result.layers)
            total_bytes = sum(l.total_bytes for l in result.layers)
            ai = total_flops / total_bytes if total_bytes > 0 else 0.0
            attainable = result.hw.attainable_performance(ai, first_dtype)
            agg[model_name].append((max(ai, 1e-3), attainable, total_flops))
            all_ai_vals.append(max(ai, 1e-3))

    # X range for roofline curves
    all_ridges = [hw.ridge_point(first_dtype) for hw in hw_list]
    ai_min = min(min(all_ai_vals) * 0.3, 1e-2)
    ai_max = max(max(all_ridges) * 10, max(all_ai_vals) * 5, 1e4)
    x_curve = np.logspace(np.log10(ai_min), np.log10(ai_max), 500)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw one roofline curve per hardware
    for hw_idx, hw in enumerate(hw_list):
        color = cmap(hw_idx % 10)
        peak_flops = hw._get_peak_flops(first_dtype)
        peak_bw = hw.peak_mem_bw
        ridge = hw.ridge_point(first_dtype)
        y_curve = np.minimum(x_curve * peak_bw, peak_flops)
        ax.plot(x_curve, y_curve, color=color, linewidth=2.2,
                label=hw.name, zorder=5)
        ax.axvline(x=ridge, color=color, linestyle="--", linewidth=0.9,
                   alpha=0.65, zorder=4)

    # Dot size scale: proportional to log(total_flops) of model (use first HW as ref)
    ref_flops = [agg[m][0][2] for m in model_names]
    max_log_flops = max(np.log1p(f) for f in ref_flops) if ref_flops else 1

    # Draw aggregate dots
    for m_idx, model_name in enumerate(model_names):
        m_marker = model_markers[m_idx % len(model_markers)]
        total_flops_ref = agg[model_name][0][2]
        dot_size = max(120, 600 * (np.log1p(total_flops_ref) / max_log_flops) ** 1.5)

        ai_positions = []
        for hw_idx, (ai, attainable, _) in enumerate(agg[model_name]):
            color = cmap(hw_idx % 10)
            ax.scatter([ai], [attainable], marker=m_marker, c=[color],
                       s=dot_size, edgecolors="white", linewidths=1.2,
                       zorder=12, alpha=0.92)
            ai_positions.append(ai)

        # Annotate model name once, at the rightmost (highest AI) HW dot position
        best_idx = int(np.argmax([a[0] for a in agg[model_name]]))
        best_ai, best_perf, _ = agg[model_name][best_idx]
        ax.annotate(
            model_name,
            xy=(best_ai, best_perf),
            xytext=(best_ai * 1.15, best_perf * 1.35),
            fontsize=9,
            fontweight="bold",
            color="#222222",
            arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.9),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#dddddd", alpha=0.9),
            zorder=20,
        )

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel("Arithmetic Intensity  (FLOPs / Byte)", fontsize=12)
    ax.set_ylabel("Attainable Performance  (FLOPs / s)", fontsize=12)
    ax.set_title(
        f"Roofline — Model vs Hardware Comparison\n"
        f"{first_dtype} | {first_mode}  "
        f"(dots = model aggregate AI; size ∝ total FLOPs)",
        fontsize=13,
    )
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v/1e12:.0f}T" if v >= 1e12
                          else f"{v/1e9:.0f}G" if v >= 1e9 else f"{v:.0e}")
    )

    # Two-section legend: hardware (color) + model (shape)
    hw_handles = [
        mlines.Line2D([], [], color=cmap(i % 10), linewidth=2.2,
                      label=hw.name)
        for i, hw in enumerate(hw_list)
    ]
    model_handles = [
        mlines.Line2D([], [], color="#555555",
                      marker=model_markers[i % len(model_markers)],
                      markersize=9, linestyle="None",
                      label=name)
        for i, name in enumerate(model_names)
    ]
    ax.legend(
        handles=hw_handles + model_handles,
        loc="upper left",
        fontsize=8.5,
        framealpha=0.88,
        ncol=2,
        title="Hardware (color)  |  Model (shape)",
        title_fontsize=8,
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[roofline] Saved: {save_path}")
    if show:
        plt.show()
    return fig


def plot_models_on_hw(
    results_dict: dict,
    figsize=(13, 8),
    save_path: Optional[str] = None,
    show: bool = True,
    log_scale: bool = True,
):
    """Draw a single-hardware roofline with one aggregate dot per model.

    Useful when the hardware is auto-detected (single target) and you want to
    compare how multiple models sit relative to each other and the hardware
    roofline ceiling, without per-layer clutter.

    Each dot represents the model as a whole:
    - X = total_flops / total_bytes  (model-level arithmetic intensity)
    - Y = attainable_performance(AI, dtype)
    - Dot size  ∝ log(total_flops)
    - Dot color = unique per model (tab10)
    - Label     = model name + total FLOPs

    Parameters
    ----------
    results_dict:
        ``{model_name: AnalysisResults}`` — all results must share the same
        hardware target (as produced when running ``analyze()`` for one HW).
    """
    if not results_dict:
        raise ValueError("results_dict must not be empty")

    try:
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
    except ImportError as e:
        raise ImportError("matplotlib is required: pip install matplotlib") from e

    # Pull HW / dtype / mode from the first result
    first = next(iter(results_dict.values()))
    hw    = first.hw
    dtype = first.dtype
    mode  = first.mode

    peak_flops = hw._get_peak_flops(dtype)
    peak_bw    = hw.peak_mem_bw
    ridge      = hw.ridge_point(dtype)

    # Compute aggregate (AI, attainable_perf, total_flops) per model
    agg = {}
    for model_name, results in results_dict.items():
        total_flops = sum(l.flops for l in results.layers)
        total_bytes = sum(l.total_bytes for l in results.layers)
        ai = total_flops / total_bytes if total_bytes > 0 else 0.0
        attainable = hw.attainable_performance(ai, dtype)
        agg[model_name] = (max(ai, 1e-3), attainable, total_flops)

    all_ai = [v[0] for v in agg.values()]
    ai_min = min(min(all_ai) * 0.3, 1e-2)
    ai_max = max(ridge * 10, max(all_ai) * 5, 1e4)
    x_curve = np.logspace(np.log10(ai_min), np.log10(ai_max), 500)
    y_curve = np.minimum(x_curve * peak_bw, peak_flops)

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("tab10")

    # Roofline curve
    ax.plot(x_curve, y_curve, color="black", linewidth=2.5,
            label="Roofline", zorder=5)
    ax.axvline(x=ridge, color="black", linestyle="--", linewidth=1.2,
               alpha=0.6, label=f"Ridge = {ridge:.1f} FLOPs/B", zorder=4)

    # Memory-bound / Compute-bound region labels
    ax.text(
        ridge * 0.28, peak_flops * 0.55,
        "Memory\nBound", ha="center", va="center",
        fontsize=10, color="#555555",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.75),
    )
    ax.text(
        ridge * 3.5, peak_flops * 0.75,
        "Compute\nBound", ha="center", va="center",
        fontsize=10, color="#555555",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.75),
    )

    # Dot size scale
    all_flops = [v[2] for v in agg.values()]
    max_log_f = max(np.log1p(f) for f in all_flops) if all_flops else 1

    model_names = list(results_dict.keys())
    handles = []

    for m_idx, model_name in enumerate(model_names):
        ai, attainable, total_flops = agg[model_name]
        color = cmap(m_idx % 10)
        dot_size = max(150, 700 * (np.log1p(total_flops) / max_log_f) ** 1.5)

        ax.scatter([ai], [attainable], c=[color], s=dot_size,
                   edgecolors="white", linewidths=1.5, zorder=12, alpha=0.92)

        flops_str = (f"{total_flops/1e9:.1f}G" if total_flops >= 1e9
                     else f"{total_flops/1e6:.1f}M")
        ax.annotate(
            f"{model_name}\n({flops_str} FLOPs)",
            xy=(ai, attainable),
            xytext=(ai * 1.35, attainable * 1.45),
            fontsize=8.5,
            color="#222222",
            arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.9),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#dddddd", alpha=0.92),
            zorder=20,
        )

        handles.append(
            mlines.Line2D([], [], color=color, marker="o", markersize=9,
                          linestyle="None", label=f"{model_name}  ({flops_str})")
        )

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel("Arithmetic Intensity  (FLOPs / Byte)", fontsize=12)
    ax.set_ylabel("Attainable Performance  (FLOPs / s)", fontsize=12)
    ax.set_title(
        f"All Models on {hw.name}\n"
        f"{dtype} | {mode}  (dot size \u221d total FLOPs)",
        fontsize=13,
    )
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v/1e12:.0f}T" if v >= 1e12
                          else f"{v/1e9:.0f}G" if v >= 1e9 else f"{v:.0e}")
    )

    roofline_handle = mlines.Line2D([], [], color="black", linewidth=2.5,
                                    label="Roofline")
    ridge_handle = mlines.Line2D([], [], color="black", linestyle="--",
                                 linewidth=1.2, label=f"Ridge ({ridge:.1f})")
    ax.legend(
        handles=[roofline_handle, ridge_handle] + handles,
        loc="upper left",
        fontsize=8.5,
        framealpha=0.88,
        title=f"Models on {hw.name}",
        title_fontsize=8,
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[roofline] Saved: {save_path}")
    if show:
        plt.show()
    return fig
