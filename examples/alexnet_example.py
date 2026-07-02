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

# -----------------------------------------------------------------
# Configuration — edit these two lines to match your setup
# -----------------------------------------------------------------
MODEL_FOLDER = r"C:\Users\keeg\OneDrive - Nokia\my_inno_work\koold\edgeAI\ROOFLINE\base_alexnet_wb\base_alexnet_wb"
HW_TARGET    = HW_DB["h100_sxm"]   # swap to: a100_80gb, rtx_4090, m3_max, …
# -----------------------------------------------------------------

results = analyze(
    model=MODEL_FOLDER,
    hw=HW_TARGET,
    dtype="float32",
    mode="inference",
)

# 1. One-line summary
results.print_summary()

# 2. Per-layer rich table
results.print_table()

# 3. Roofline plot (saved to current directory)
results.plot_roofline(save_path="alexnet_roofline.png", show=True)

# 4. Pandas DataFrame for custom analysis
df = results.to_dataframe()
print("\nDataFrame preview:")
print(df[["name", "type", "flops_fmt", "arithmetic_intensity", "bottleneck"]])
