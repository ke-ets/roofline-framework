"""TableReport — tabular display of per-layer roofline analysis results."""

from __future__ import annotations

from typing import Optional

from roofline.core.layer_info import LayerStats


class _ListShim:
    """Minimal shim so TableReport works when initialised with a bare list of LayerStats."""

    def __init__(self, layers):
        self.layers = layers
        self.hw = None
        self.dtype = "unknown"

    def summary(self) -> str:
        total = len(self.layers)
        nb = sum(1 for l in self.layers if l.bottleneck == "memory")
        cb = total - nb
        total_flops = sum(l.flops for l in self.layers)
        return (
            f"Layer report — {total} layers  |  "
            f"FLOPs: {total_flops:.3e}  |  "
            f"Memory-bound: {nb}  Compute-bound: {cb}"
        )


class TableReport:
    """Renders ``AnalysisResults`` as a pandas DataFrame or a rich terminal table."""

    def __init__(self, results):
        # Accept either a full AnalysisResults object or a bare list of LayerStats.
        # When given a plain list, wrap it in a lightweight shim so all downstream
        # code can always call self.results.layers, self.results.summary(), etc.
        if isinstance(results, list):
            results = _ListShim(results)
        self.results = results

    # ------------------------------------------------------------------
    # DataFrame
    # ------------------------------------------------------------------

    def to_dataframe(self):
        """Return a pandas DataFrame with one row per layer."""
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas is required: pip install pandas") from e

        rows = [ls.summary_dict() for ls in self.results.layers]
        df = pd.DataFrame(rows)

        # Human-readable columns
        if "flops" in df.columns:
            df["flops_fmt"] = df["flops"].apply(_fmt_flops)
        if "total_bytes" in df.columns:
            df["total_bytes_fmt"] = df["total_bytes"].apply(_fmt_bytes)
        if "params" in df.columns:
            df["params_fmt"] = df["params"].apply(_fmt_params)
        if "energy_j" in df.columns:
            df["energy_uj"] = df["energy_j"].apply(lambda v: f"{v*1e6:.4f}")
        if "energy_efficiency" in df.columns:
            df["energy_eff_gflops_j"] = df["energy_efficiency"].apply(lambda v: f"{v/1e9:.4f}")

        return df

    def to_csv(self, path: str) -> None:
        """Save the DataFrame to a CSV file."""
        self.to_dataframe().to_csv(path, index=False)
        print(f"[roofline] Saved report to {path}")

    # ------------------------------------------------------------------
    # Rich terminal table
    # ------------------------------------------------------------------

    def print_rich_table(
        self,
        top_n: Optional[int] = None,
        sort_by: str = "flops",
    ) -> None:
        """Print a formatted table using the ``rich`` library."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich import box
        except ImportError as e:
            raise ImportError("rich is required: pip install rich") from e

        layers = self.results.layers
        if sort_by == "flops":
            layers = sorted(layers, key=lambda l: l.flops, reverse=True)
        elif sort_by == "time":
            layers = sorted(layers, key=lambda l: l.theoretical_time_ms, reverse=True)
        elif sort_by == "ai":
            layers = sorted(layers, key=lambda l: l.arithmetic_intensity, reverse=True)
        elif sort_by == "bytes":
            layers = sorted(layers, key=lambda l: l.total_bytes, reverse=True)

        if top_n:
            layers = layers[:top_n]

        console = Console()

        # Summary header
        console.print(f"\n[bold cyan]{self.results.summary()}[/bold cyan]\n")

        hw_label = self.results.hw.name if self.results.hw else "unknown HW"
        table = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            title=f"Layer-wise Roofline Analysis — {hw_label} [{self.results.dtype}]",
        )

        table.add_column("Layer", style="dim", max_width=40, no_wrap=False)
        table.add_column("Type", style="cyan", max_width=20)
        table.add_column("Params", justify="right")
        table.add_column("FLOPs", justify="right", style="yellow")
        table.add_column("Weights", justify="right")
        table.add_column("Activations", justify="right")
        table.add_column("AI\n(FLOPs/B)", justify="right", style="green")
        table.add_column("Attainable\n(TFLOPs)", justify="right")
        table.add_column("Bottleneck", justify="center")
        table.add_column("Time (ms)", justify="right", style="blue")
        table.add_column("Energy (µJ)", justify="right", style="magenta")
        table.add_column("Effic.\n(GFLOPS/J)", justify="right", style="magenta")

        for ls in layers:
            bottleneck_style = "[red]memory[/red]" if ls.bottleneck == "memory" else "[green]compute[/green]"
            table.add_row(
                ls.layer.name,
                ls.layer.layer_type,
                _fmt_params(ls.layer.num_params),
                _fmt_flops(ls.flops),
                _fmt_bytes(ls.weight_bytes),
                _fmt_bytes(ls.activation_bytes),
                f"{ls.arithmetic_intensity:.2f}",
                f"{ls.attainable_perf/1e12:.3f}",
                bottleneck_style,
                f"{ls.theoretical_time_ms:.4f}",
                f"{ls.energy_j*1e6:.4f}",
                f"{ls.energy_efficiency/1e9:.4f}",
            )

        console.print(table)

        # Footer totals
        total_flops = sum(layer.flops for layer in self.results.layers)
        total_bytes = sum(layer.total_bytes for layer in self.results.layers)
        total_time = sum(layer.theoretical_time_ms for layer in self.results.layers)
        total_energy_j = sum(layer.energy_j for layer in self.results.layers)
        total_eff = total_flops / total_energy_j if total_energy_j > 0 else 0.0
        console.print(
            f"\n[bold]Totals:[/bold] "
            f"FLOPs={_fmt_flops(total_flops)}  "
            f"Memory={_fmt_bytes(total_bytes)}  "
            f"Est. Time={total_time:.3f} ms  "
            f"Energy={total_energy_j*1e6:.3f} µJ  "
            f"Effic.={total_eff/1e9:.3f} GFLOPS/J\n"
        )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_flops(n: int) -> str:
    if n >= 1e12:
        return f"{n/1e12:.3f}T"
    if n >= 1e9:
        return f"{n/1e9:.2f}G"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.0f}K"
    return str(n)


def _fmt_bytes(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f}GB"
    if n >= 1e6:
        return f"{n/1e6:.1f}MB"
    if n >= 1e3:
        return f"{n/1e3:.0f}KB"
    return f"{n}B"


def _fmt_params(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.0f}K"
    return str(n)
