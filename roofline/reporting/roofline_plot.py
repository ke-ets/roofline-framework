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
    "MaxPool": "#8C8C8C",
    "AvgPool": "#8C8C8C",
    "AdaptiveAvgPool": "#8C8C8C",
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
        ax.set_title(
            f"Roofline Model — {hw.name}\n"
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
