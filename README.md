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
