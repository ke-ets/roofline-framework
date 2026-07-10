# Roofline Analysis Framework

Layer-wise **arithmetic intensity**, **FLOPs**, and **memory traffic** estimation for deep learning models, with hardware roofline analysis. Accepts PyTorch, ONNX, HuggingFace, TF/Keras models — as objects, files, folders, or zip archives.

---

## Quick Start

```python
from roofline import analyze, HW_DB

import torch.nn as nn
model = nn.Sequential(nn.Linear(784, 2048), nn.GELU(), nn.LayerNorm(2048), nn.Linear(2048, 10))

results = analyze(
    model=model,
    input_shapes=[(1, 784)],
    hw=HW_DB["h100_sxm"],
    dtype="float16",
)

results.print_table()       # rich terminal table
results.plot_roofline()     # matplotlib roofline chart
df = results.to_dataframe() # pandas DataFrame
print(results.summary())
```

---

## Installation

```bash
pip install -e ".[all]"   # all optional backends
pip install -e ".[dev]"   # + pytest
```

Core dependencies (always installed): `torch`, `numpy`, `pandas`, `matplotlib`, `rich`, `typer`, `requests`, `beautifulsoup4`

Optional extras:
| Extra | Packages |
|---|---|
| `onnx` | `onnx`, `onnxruntime` |
| `huggingface` | `transformers`, `safetensors` |
| `tensorflow` | `tensorflow` |
| `nvidia` | `pynvml` |
| `notebook` | `jupyter`, `ipywidgets` |

---

## Model Input Formats

| Input | How to pass |
|---|---|
| `torch.nn.Module` | Pass directly |
| ONNX file | `model="path/to/model.onnx"` |
| HuggingFace model | `model="bert-base-uncased"` |
| TF/Keras model | Pass `tf.keras.Model` object or `.h5` path |
| Local HF save | `model="./my_model_dir/"` (has `config.json`) |
| Manual layer folder | `model="./layers/"` (`.pt`, `.safetensors`, etc.) |
| Zip archive | `model="./model.zip"` |

The source type is auto-detected from the model type / file extension. Override with `source="pytorch"` etc.

---

## Hardware Targets

```python
from roofline import HW_DB
print(list(HW_DB.keys()))
```

Built-in targets:

| Vendor | Keys |
|---|---|
| NVIDIA | `v100_sxm`, `a100_40gb`, `a100_80gb`, `h100_sxm`, `h100_pcie`, `h200_sxm`, `rtx_3090`, `rtx_4090`, `rtx_5090` |
| AMD | `mi300x` |
| Intel | `gaudi2`, `gaudi3` |
| Apple | `m2`, `m2_pro`, `m2_max`, `m2_ultra`, `m3`, `m3_pro`, `m3_max`, `m3_ultra`, `m4`, `m4_pro`, `m4_max`, `m4_ultra`, `m5`, `m5_pro`, `m5_max`, `m5_ultra` |

### Auto-detect local GPU

```python
from roofline import detect_hw
hw = detect_hw()   # NVIDIA via pynvml, AMD via rocm-smi, Apple via sysctl
```

### Fetch unknown hardware from the web

```python
from roofline import fetch_hw
hw = fetch_hw("RTX 5090", fetch_from_web=True)  # scrapes TechPowerUp, caches to ~/.roofline/hw_cache.json
```

---

## Analysis Modes

| Mode | Memory traffic |
|---|---|
| `"inference"` | weights + activations |
| `"training"` | weights + activations + gradients + Adam optimizer states |

---

## CLI

```bash
# List built-in hardware
roofline list-hw
roofline list-hw --vendor nvidia

# Analyse a model
roofline analyze bert-base-uncased --hw h100_sxm --dtype float16 --source huggingface

# Auto-detect local GPU
roofline analyze resnet50.onnx --detect-hw --dtype float32

# Fetch specs for unlisted hardware (prompts for confirmation)
roofline analyze ./my_model.pt --fetch-hw "RTX 5090" --dtype bfloat16

# Save outputs
roofline analyze ./model.onnx --hw rtx_4090 --output report.csv --plot roofline.png

# Show layer info without hardware analysis
roofline info bert-base-uncased --source huggingface --input-shape 1,128
```

---

## Examples

### AlexNet — folder-based input across 4 hardware targets

Analyses a locally-saved AlexNet model (directory of per-layer `.pt` weight files)
against Raspberry Pi 4, Raspberry Pi 5, Apple M4, and Arduino Nicla Vision.

**1. Edit the model folder path in the script:**

```python
# examples/alexnet_example.py
MODEL_FOLDER = r"/path/to/base_alexnet_wb"   # directory with .pt weight files
```

**2. Run:**

```bash
python examples/alexnet_example.py
```

**What it produces:**
- Console output — hardware specs, one-line summary, and a full layer-by-layer breakdown table for each hardware target
- `alexnet_multi_hw_roofline.png` — all 4 HW roofline curves overlaid with the AlexNet layer scatter

---

### Multi-Model × Multi-Hardware

Runs roofline analysis on **AlexNet, MobileNetV2, ResNet101, VGG16** across the same 4 hardware targets using live `torchvision` model objects (no pre-saved weights needed).

**Prerequisites:**

```bash
pip install torchvision
```

**Run:**

```bash
python examples/multi_model_multi_hw.py
```

**What it produces:**

| File | Contents |
|---|---|
| `alexnet_multi_hw_roofline.png` | AlexNet layers on all 4 HW — dot fill = layer type, dot border/shape = HW |
| `mobilenetv2_multi_hw_roofline.png` | MobileNetV2 layers on all 4 HW |
| `resnet101_multi_hw_roofline.png` | ResNet101 layers on all 4 HW |
| `vgg16_multi_hw_roofline.png` | VGG16 layers on all 4 HW |
| `multi_model_multi_hw_roofline.png` | Unified aggregate view — 1 dot per (model, HW) pair |

**Per-model plots — dual-legend design:**
- **Upper-left "Hardware"** legend — roofline curve color + dot border color + marker shape per HW
- **Lower-right "Layer types"** legend — dot fill color per layer type (Conv2d, Linear, BatchNorm, ReLU, MaxPool, AdaptiveAvgPool, Dropout, …)

**Unified aggregate plot — what to read:**
- Each model forms a **vertical cluster of 4 same-shaped dots** at the model's overall arithmetic intensity (AI = total FLOPs / total bytes — hardware-independent)
- The 4 hardware-coloured dots in each cluster show how each hardware's roofline ceiling clips attainable performance; a dot sitting on the flat part of a curve is compute-bound, a dot on the rising slope is memory-bound

---

### HW-Detect Experiment (7 models on auto-detected hardware)

Auto-detects the local GPU/CPU, then runs roofline analysis on 4 torchvision CNNs and 3 HuggingFace language models against the detected hardware.

**No manual setup needed** — missing packages (`torchvision`, `transformers`, `huggingface_hub`) are installed automatically on first run.

```bash
python examples/detect_hw_roofline.py
```

Hardware detection order: NVIDIA GPU (pynvml) → AMD GPU (rocm-smi) → Apple Silicon (sysctl) → CPU fallback. If the detected chip is not in the built-in database, the script attempts a web fetch from TechPowerUp; if that also fails it falls back to Raspberry Pi 5 as a reference.

**Models analyzed:**

| Model | Source | Notes |
|---|---|---|
| AlexNet | torchvision | CNN |
| ResNet101 | torchvision | CNN |
| VGG16 | torchvision | CNN |
| MobileNetV2 | torchvision | Lightweight CNN |
| GPT-2 | HuggingFace `openai-community/gpt2` | Transformer LM |
| DeepSeek-Coder 1.3B | HuggingFace `deepseek-ai/deepseek-coder-1.3b-base` | Code LM |
| Gemma 2B | HuggingFace `google/gemma-2b` | Requires `huggingface-cli login` |

**What it produces:**
- 7 per-model layer-wise roofline plots (e.g. `alexnet_roofline.png`)
- 1 consolidated plot `all_models_on_detected_hw.png` — single roofline curve for the detected hardware with one aggregate dot per model (dot size ∝ total FLOPs), showing how each model's compute-intensity compares against the hardware ceiling

> **Note:** Gemma 2B is a gated model. If you see an authentication error, run `huggingface-cli login` first. The script skips Gemma gracefully and proceeds with the remaining models.

---

## Roofline Model

For each layer and hardware target:

```
ridge_point         = peak_flops[dtype] / peak_mem_bw
arithmetic_intensity = FLOPs / (weight_bytes + activation_bytes)
attainable_perf     = min(AI × peak_mem_bw, peak_flops[dtype])
bottleneck          = "compute" if AI > ridge_point else "memory"
theoretical_time_ms = FLOPs / attainable_perf × 1000
```

---

## Energy Efficiency

For each layer and the model as a whole, the framework estimates energy consumption and energy efficiency (FLOPS/J) using a TDP-based two-term model:

```
E_per_flop  = TDP × 0.75 / peak_flops[dtype]   # J / FLOP  (75% of TDP to compute)
E_per_byte  = TDP × 0.25 / peak_mem_bw          # J / byte  (25% of TDP to memory)
E_total     = E_per_flop × FLOPs + E_per_byte × bytes   # Joules
efficiency  = FLOPs / E_total  =  AI / (E_per_flop × AI + E_per_byte)   # FLOPS/J
```

The efficiency equals `FLOPs/s ÷ average_power` — the two forms are algebraically identical.
Ceiling (compute-bound, AI → ∞): `1 / E_per_flop`.  Low-AI slope (memory-bound): `AI / E_per_byte`.

The 75/25 split is adjustable via `HWSpec.COMPUTE_POWER_FRACTION` / `MEMORY_POWER_FRACTION`.

### Energy plot functions

```python
from roofline.reporting.roofline_plot import (
    plot_model_across_hw_energy,      # per-model: perf roofline (left) + FLOPS/J curve (right)
    plot_multi_model_multi_hw_energy, # combined: all models × all HW, dual-axis
)
```

See `examples/energy_efficiency_multi_model.py` for a full demo (4 models × 4 hardware targets → 5 plots).

---

## Extending with Custom FLOPs Handlers

```python
from roofline.core.analyzer import RooflineAnalyzer

analyzer = RooflineAnalyzer()

@analyzer.flop_counter.register("MyCustomOp")
def count_my_op(layer):
    return layer.attrs.get("n_elements", 0) * 3  # 3 ops per element
```

---

## Project Structure

```
roofline/
├── __init__.py              # analyze(), HW_DB, detect_hw(), fetch_hw()
├── parsers/
│   ├── pytorch_parser.py    # torch.fx + hooks
│   ├── onnx_parser.py       # onnx graph walker
│   ├── huggingface_parser.py
│   ├── tensorflow_parser.py
│   ├── folder_parser.py     # auto-detects HF-local vs manual layer files
│   └── zip_handler.py       # extracts → FolderParser → cleanup
├── core/
│   ├── layer_info.py        # LayerInfo, LayerStats dataclasses
│   ├── flop_counter.py      # handler registry per op type
│   ├── memory_estimator.py  # weight + activation bytes
│   └── analyzer.py          # RooflineAnalyzer, AnalysisResults
├── hardware/
│   ├── hw_spec.py           # HWSpec dataclass
│   ├── hw_database.py       # 28-entry built-in registry
│   ├── hw_detector.py       # pynvml / rocm-smi / sysctl detection
│   ├── hw_fetcher.py        # TechPowerUp web scraper + cache
│   └── hw_resolver.py       # 4-step resolution cascade
├── reporting/
│   ├── table_report.py      # pandas + rich table
│   └── roofline_plot.py     # matplotlib roofline chart
└── cli.py                   # typer CLI
examples/
├── alexnet_example.py       # single-model multi-HW (folder input)
├── multi_model_multi_hw.py  # 4 CNN models × 4 HW → 5 plots
└── detect_hw_roofline.py    # auto-detect local HW → 7 models → 8 plots
notebooks/
└── demo.ipynb               # end-to-end walkthrough
tests/
├── test_flop_counter.py
├── test_memory_estimator.py
└── test_roofline.py
```

---

## Running Tests

```bash
pytest tests/ -v
```
