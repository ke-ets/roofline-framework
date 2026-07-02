"""CLI entry point for the Roofline Analysis Framework.

Usage examples
--------------
# Use a built-in hardware target
roofline analyze --model bert-base-uncased --source huggingface \\
                 --hw h100_sxm --dtype float16

# Auto-detect local GPU
roofline analyze --model resnet50.onnx --detect-hw

# Fetch specs for a hardware not in the built-in DB
roofline analyze --model ./my_model.pt --fetch-hw "RTX 5090" \\
                 --dtype bfloat16 --output report.csv --plot roofline.png

# List all built-in hardware targets
roofline list-hw

# Show a summary of a model without hardware analysis
roofline info --model bert-base-uncased --source huggingface
"""

from __future__ import annotations

from typing import List, Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="roofline",
    help="Layer-wise roofline analysis for deep learning models.",
    add_completion=False,
)
console = Console()


@app.command()
def analyze(
    model: str = typer.Argument(..., help="Model path, file, or HuggingFace model name."),
    source: Optional[str] = typer.Option(
        None, "--source", "-s",
        help="Source type: pytorch | onnx | huggingface | tensorflow | folder | zip. Auto-detected if omitted.",
    ),
    hw: Optional[str] = typer.Option(
        None, "--hw",
        help="Hardware key from the built-in database (e.g. h100_sxm, rtx_4090, m4_pro). "
             "Run `roofline list-hw` to see all options.",
    ),
    detect_hw: bool = typer.Option(
        False, "--detect-hw",
        help="Auto-detect the GPU/accelerator on this machine.",
    ),
    fetch_hw: Optional[str] = typer.Option(
        None, "--fetch-hw",
        help="Fetch specs for a named hardware from TechPowerUp (will prompt for approval).",
    ),
    dtype: str = typer.Option(
        "float32", "--dtype", "-d",
        help="Precision: float32 | float16 | bfloat16 | int8 | int4.",
    ),
    mode: str = typer.Option(
        "inference", "--mode", "-m",
        help="Analysis mode: inference | training.",
    ),
    input_shape: Optional[List[str]] = typer.Option(
        None, "--input-shape",
        help="Input shape as comma-separated ints, e.g. '1,3,224,224'. Repeat for multiple inputs.",
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", "-b",
        help="Override batch dimension in all input shapes.",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Save layer-wise results to this CSV file path.",
    ),
    plot: Optional[str] = typer.Option(
        None, "--plot", "-p",
        help="Save roofline plot to this image path (.png, .pdf, .svg).",
    ),
    top_n: Optional[int] = typer.Option(
        None, "--top-n",
        help="Show only the top N layers by FLOPs in the terminal table.",
    ),
    sort_by: str = typer.Option(
        "flops", "--sort-by",
        help="Sort table by: flops | time | ai | bytes.",
    ),
    no_table: bool = typer.Option(
        False, "--no-table",
        help="Skip printing the terminal table.",
    ),
):
    """Analyse a model's layer-wise arithmetic intensity and roofline bounds."""
    from roofline.core.analyzer import RooflineAnalyzer

    # ---- Parse input shapes ----
    parsed_shapes = None
    if input_shape:
        parsed_shapes = []
        for s in input_shape:
            try:
                parsed_shapes.append(tuple(int(x.strip()) for x in s.split(",")))
            except ValueError:
                console.print(f"[red]Invalid input shape '{s}'. Use comma-separated ints like '1,3,224,224'.[/red]")
                raise typer.Exit(1)

    # ---- Resolve hardware ----
    hw_spec = None

    if hw:
        from roofline.hardware.hw_database import lookup
        hw_spec = lookup(hw)
        if hw_spec is None:
            console.print(f"[red]Hardware '{hw}' not found in built-in DB. Run `roofline list-hw` to see options.[/red]")
            raise typer.Exit(1)

    elif fetch_hw:
        # Ask for confirmation before web fetch
        confirmed = typer.confirm(f"Fetch specs for '{fetch_hw}' from TechPowerUp?", default=False)
        if not confirmed:
            console.print("[yellow]Fetch cancelled. Use --hw to pick from the built-in database.[/yellow]")
            raise typer.Exit(0)
        from roofline.hardware.hw_fetcher import fetch_hw as _fetch
        try:
            hw_spec = _fetch(fetch_hw, fetch_from_web=True)
        except Exception as e:
            console.print(f"[red]Failed to fetch specs: {e}[/red]")
            raise typer.Exit(1)

    elif detect_hw:
        from roofline.hardware.hw_detector import detect_hw as _detect, HWDetectionError
        try:
            hw_spec = _detect()
        except HWDetectionError as e:
            console.print(f"[yellow]{e}[/yellow]")
            # Offer web fetch
            detected_name = str(e).split("'")[1] if "'" in str(e) else None
            if detected_name:
                confirmed = typer.confirm(f"Fetch specs for '{detected_name}' from the web?", default=False)
                if confirmed:
                    from roofline.hardware.hw_fetcher import fetch_hw as _fetch
                    hw_spec = _fetch(detected_name, fetch_from_web=True)
                else:
                    raise typer.Exit(1)
            else:
                raise typer.Exit(1)

    # ---- Run analysis ----
    analyzer = RooflineAnalyzer()
    try:
        results = analyzer.analyze(
            model=model,
            input_shapes=parsed_shapes,
            hw=hw_spec,
            dtype=dtype,
            mode=mode,
            source=source,
            batch_size=batch_size,
            fetch_from_web=False,  # already handled above
            detect_local_hw=detect_hw and hw_spec is None,
        )
    except Exception as e:
        console.print(f"[red]Analysis failed: {e}[/red]")
        raise typer.Exit(1)

    # ---- Output ----
    if not no_table:
        results.print_table(top_n=top_n, sort_by=sort_by)

    if output:
        from roofline.reporting.table_report import TableReport
        TableReport(results).to_csv(output)

    if plot:
        results.plot_roofline(save_path=plot, show=False)


@app.command("list-hw")
def list_hw(
    vendor: Optional[str] = typer.Option(None, "--vendor", "-v", help="Filter by vendor: nvidia | amd | intel | apple."),
    verbose: bool = typer.Option(False, "--verbose", "-V", help="Show full specs."),
):
    """List all built-in hardware targets."""
    from roofline.hardware.hw_database import HW_DB
    from rich.table import Table
    from rich import box

    table = Table(box=box.ROUNDED, title="Built-in Hardware Database", header_style="bold magenta")
    table.add_column("Key", style="cyan")
    table.add_column("Name")
    table.add_column("Vendor")
    table.add_column("Architecture")
    table.add_column("FP16 TFLOPs", justify="right")
    table.add_column("Mem BW (TB/s)", justify="right")
    table.add_column("VRAM (GB)", justify="right")
    table.add_column("TDP (W)", justify="right")

    for key, spec in HW_DB.items():
        if vendor and spec.vendor.lower() != vendor.lower():
            continue
        fp16 = spec.peak_flops.get("float16") or spec.peak_flops.get("bfloat16") or 0
        table.add_row(
            key,
            spec.name,
            spec.vendor.upper(),
            spec.architecture,
            f"{fp16/1e12:.0f}",
            f"{spec.peak_mem_bw/1e12:.2f}",
            f"{spec.memory_capacity_gb:.0f}",
            f"{spec.tdp_watts:.0f}",
        )

    console.print(table)


@app.command("info")
def model_info(
    model: str = typer.Argument(..., help="Model path, file, or HuggingFace model name."),
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Source type."),
    input_shape: Optional[List[str]] = typer.Option(None, "--input-shape"),
    dtype: str = typer.Option("float32", "--dtype"),
):
    """Show layer info for a model without hardware analysis."""
    from roofline.core.analyzer import _infer_source, RooflineAnalyzer
    from rich.table import Table
    from rich import box

    parsed_shapes = None
    if input_shape:
        parsed_shapes = [tuple(int(x) for x in s.split(",")) for s in input_shape]

    src = source or _infer_source(model)
    analyzer = RooflineAnalyzer()
    try:
        layers = analyzer._parse(model, src, parsed_shapes, dtype)
    except Exception as e:
        console.print(f"[red]Parsing failed: {e}[/red]")
        raise typer.Exit(1)

    table = Table(box=box.ROUNDED, title=f"Layer Info — {model}", header_style="bold magenta")
    table.add_column("Layer", max_width=40)
    table.add_column("Type", style="cyan")
    table.add_column("Params", justify="right")
    table.add_column("Input Shapes")
    table.add_column("Output Shapes")
    table.add_column("Dtype")

    total_params = 0
    for li in layers:
        total_params += li.num_params
        table.add_row(
            li.name,
            li.layer_type,
            _fmt_params(li.num_params),
            str(li.input_shapes),
            str(li.output_shapes),
            li.dtype,
        )

    console.print(table)
    console.print(f"\n[bold]Total parameters: {_fmt_params(total_params)}[/bold]\n")


def _fmt_params(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.0f}K"
    return str(n)


if __name__ == "__main__":
    app()
