"""Built-in hardware registry.

All specs are sourced from official datasheets / product pages:
  NVIDIA  — https://www.nvidia.com/en-us/data-center/
  AMD     — https://www.amd.com/en/products/accelerators/
  Intel   — https://www.intel.com/content/www/us/en/products/sku/
  Apple   — https://www.apple.com/mac/

Access entries via ``HW_DB["key"]`` or list all keys with ``list(HW_DB)``.
"""

from __future__ import annotations

from typing import Dict

from roofline.hardware.hw_spec import HWSpec

# ---------------------------------------------------------------------------
# NVIDIA — Data Centre
# ---------------------------------------------------------------------------

_NVIDIA_V100_SXM = HWSpec(
    name="NVIDIA V100 SXM2 32GB",
    vendor="nvidia",
    architecture="Volta",
    peak_flops={
        "float64": 7.8e12,
        "float32": 15.7e12,
        "float16": 125.0e12,
    },
    peak_mem_bw=0.9e12,       # 900 GB/s HBM2
    memory_capacity_gb=32,
    memory_type="HBM2",
    tdp_watts=300,
    aliases=("v100", "v100_sxm", "v100 sxm2"),
)

_NVIDIA_A100_40 = HWSpec(
    name="NVIDIA A100 SXM 40GB",
    vendor="nvidia",
    architecture="Ampere",
    peak_flops={
        "float64": 19.5e12,
        "float32": 19.5e12,
        "bfloat16": 312.0e12,
        "float16": 312.0e12,
        "int8": 624.0e12,
    },
    peak_mem_bw=1.555e12,     # 1555 GB/s HBM2e
    memory_capacity_gb=40,
    memory_type="HBM2e",
    tdp_watts=400,
    aliases=("a100", "a100_40", "a100 sxm 40gb", "a100-sxm-40"),
)

_NVIDIA_A100_80 = HWSpec(
    name="NVIDIA A100 SXM 80GB",
    vendor="nvidia",
    architecture="Ampere",
    peak_flops={
        "float64": 19.5e12,
        "float32": 19.5e12,
        "bfloat16": 312.0e12,
        "float16": 312.0e12,
        "int8": 624.0e12,
    },
    peak_mem_bw=2.0e12,       # 2 TB/s HBM2e
    memory_capacity_gb=80,
    memory_type="HBM2e",
    tdp_watts=400,
    aliases=("a100_80", "a100 sxm 80gb", "a100-sxm-80"),
)

_NVIDIA_H100_SXM = HWSpec(
    name="NVIDIA H100 SXM5 80GB",
    vendor="nvidia",
    architecture="Hopper",
    peak_flops={
        "float64": 33.5e12,
        "float32": 67.0e12,
        "bfloat16": 989.0e12,
        "float16": 989.0e12,
        "int8": 1979.0e12,
        "int4": 3958.0e12,
    },
    peak_mem_bw=3.35e12,      # 3.35 TB/s HBM3
    memory_capacity_gb=80,
    memory_type="HBM3",
    tdp_watts=700,
    aliases=("h100", "h100_sxm", "h100 sxm", "h100-sxm"),
)

_NVIDIA_H100_PCIE = HWSpec(
    name="NVIDIA H100 PCIe 80GB",
    vendor="nvidia",
    architecture="Hopper",
    peak_flops={
        "float64": 26.0e12,
        "float32": 51.0e12,
        "bfloat16": 756.0e12,
        "float16": 756.0e12,
        "int8": 1513.0e12,
        "int4": 3026.0e12,
    },
    peak_mem_bw=2.0e12,       # 2 TB/s HBM2e
    memory_capacity_gb=80,
    memory_type="HBM2e",
    tdp_watts=350,
    aliases=("h100_pcie", "h100 pcie", "h100-pcie"),
)

_NVIDIA_H200_SXM = HWSpec(
    name="NVIDIA H200 SXM 141GB",
    vendor="nvidia",
    architecture="Hopper",
    peak_flops={
        "float64": 33.5e12,
        "float32": 67.0e12,
        "bfloat16": 989.0e12,
        "float16": 989.0e12,
        "int8": 1979.0e12,
        "int4": 3958.0e12,
    },
    peak_mem_bw=4.8e12,       # 4.8 TB/s HBM3e
    memory_capacity_gb=141,
    memory_type="HBM3e",
    tdp_watts=700,
    aliases=("h200", "h200_sxm", "h200 sxm"),
)

_NVIDIA_RTX_3090 = HWSpec(
    name="NVIDIA GeForce RTX 3090",
    vendor="nvidia",
    architecture="Ampere",
    peak_flops={
        "float32": 35.6e12,
        "float16": 71.2e12,
        "int8": 142.0e12,
    },
    peak_mem_bw=0.936e12,     # 936 GB/s GDDR6X
    memory_capacity_gb=24,
    memory_type="GDDR6X",
    tdp_watts=350,
    aliases=("rtx3090", "rtx 3090", "rtx_3090", "3090"),
)

_NVIDIA_RTX_4090 = HWSpec(
    name="NVIDIA GeForce RTX 4090",
    vendor="nvidia",
    architecture="Ada Lovelace",
    peak_flops={
        "float32": 82.6e12,
        "float16": 165.2e12,
        "int8": 330.4e12,
    },
    peak_mem_bw=1.008e12,     # 1008 GB/s GDDR6X
    memory_capacity_gb=24,
    memory_type="GDDR6X",
    tdp_watts=450,
    aliases=("rtx4090", "rtx 4090", "rtx_4090", "4090"),
)

_NVIDIA_RTX_5090 = HWSpec(
    name="NVIDIA GeForce RTX 5090",
    vendor="nvidia",
    architecture="Blackwell",
    peak_flops={
        "float32": 104.8e12,
        "float16": 838.0e12,
        "bfloat16": 838.0e12,
        "int8": 1676.0e12,
    },
    peak_mem_bw=1.79e12,      # 1792 GB/s GDDR7
    memory_capacity_gb=32,
    memory_type="GDDR7",
    tdp_watts=575,
    aliases=("rtx5090", "rtx 5090", "rtx_5090", "5090"),
)

# ---------------------------------------------------------------------------
# AMD
# ---------------------------------------------------------------------------

_AMD_MI300X = HWSpec(
    name="AMD Instinct MI300X",
    vendor="amd",
    architecture="CDNA3",
    peak_flops={
        "float64": 163.4e12,
        "float32": 163.4e12,
        "bfloat16": 1307.0e12,
        "float16": 1307.0e12,
        "int8": 2614.0e12,
        "int4": 5228.0e12,
    },
    peak_mem_bw=5.3e12,       # 5.3 TB/s HBM3
    memory_capacity_gb=192,
    memory_type="HBM3",
    tdp_watts=750,
    aliases=("mi300x", "mi300", "mi300 x", "instinct mi300x"),
)

# ---------------------------------------------------------------------------
# Intel
# ---------------------------------------------------------------------------

_INTEL_GAUDI2 = HWSpec(
    name="Intel Gaudi2",
    vendor="intel",
    architecture="Gaudi2",
    peak_flops={
        "bfloat16": 432.0e12,
        "float16": 432.0e12,
        "int8": 865.0e12,
    },
    peak_mem_bw=2.45e12,      # 2.45 TB/s HBM2e
    memory_capacity_gb=96,
    memory_type="HBM2e",
    tdp_watts=600,
    aliases=("gaudi2", "gaudi 2", "intel gaudi2"),
)

_INTEL_GAUDI3 = HWSpec(
    name="Intel Gaudi3",
    vendor="intel",
    architecture="Gaudi3",
    peak_flops={
        "float32": 1835.0e12,
        "bfloat16": 1835.0e12,
        "float16": 1835.0e12,
        "int8": 3670.0e12,
    },
    peak_mem_bw=3.7e12,       # 3.7 TB/s HBM2e
    memory_capacity_gb=128,
    memory_type="HBM2e",
    tdp_watts=900,
    aliases=("gaudi3", "gaudi 3", "intel gaudi3"),
)

# ---------------------------------------------------------------------------
# Apple Silicon — all tiers, all generations
# Specs: Apple product pages + Anandtech / TechPowerUp
# GPU FP32 TFLOPS estimated from GPU core count × clock × 2 (FMA) × cores
# Memory bandwidth from Apple specs
# ---------------------------------------------------------------------------

def _apple(name, arch, gpu_fp32_tflops, gpu_fp16_tflops, mem_bw_gbs,
           mem_gb, tdp, ane_tops, aliases):
    return HWSpec(
        name=name,
        vendor="apple",
        architecture=arch,
        peak_flops={
            "float32": gpu_fp32_tflops * 1e12,
            "float16": gpu_fp16_tflops * 1e12,
            "bfloat16": gpu_fp16_tflops * 1e12,
            "int8": ane_tops * 1e12,
        },
        peak_mem_bw=mem_bw_gbs * 1e9,
        memory_capacity_gb=mem_gb,
        memory_type="LPDDR5X",
        tdp_watts=tdp,
        unified_memory=True,
        ane_tops=ane_tops,
        aliases=aliases,
    )


# M2 family
_APPLE_M2 = _apple(
    "Apple M2", "M2", 3.6, 7.2, 100, 24, 25, 15.8,
    ("m2", "apple m2"),
)
_APPLE_M2_PRO = _apple(
    "Apple M2 Pro", "M2", 6.79, 13.6, 200, 32, 40, 15.8,
    ("m2 pro", "m2pro", "apple m2 pro"),
)
_APPLE_M2_MAX = _apple(
    "Apple M2 Max", "M2", 13.6, 27.2, 400, 96, 60, 15.8,
    ("m2 max", "m2max", "apple m2 max"),
)
_APPLE_M2_ULTRA = _apple(
    "Apple M2 Ultra", "M2", 27.2, 54.4, 800, 192, 120, 31.6,
    ("m2 ultra", "m2ultra", "apple m2 ultra"),
)

# M3 family
_APPLE_M3 = _apple(
    "Apple M3", "M3", 3.6, 7.2, 100, 24, 22, 18.0,
    ("m3", "apple m3"),
)
_APPLE_M3_PRO = _apple(
    "Apple M3 Pro", "M3", 7.0, 14.0, 150, 36, 30, 18.0,
    ("m3 pro", "m3pro", "apple m3 pro"),
)
_APPLE_M3_MAX = _apple(
    "Apple M3 Max", "M3", 14.2, 28.4, 400, 128, 50, 18.0,
    ("m3 max", "m3max", "apple m3 max"),
)
_APPLE_M3_ULTRA = _apple(
    "Apple M3 Ultra", "M3", 28.4, 56.8, 800, 192, 100, 36.0,
    ("m3 ultra", "m3ultra", "apple m3 ultra"),
)

# M4 family
_APPLE_M4 = _apple(
    "Apple M4", "M4", 4.6, 9.2, 120, 32, 25, 38.0,
    ("m4", "apple m4"),
)
_APPLE_M4_PRO = _apple(
    "Apple M4 Pro", "M4", 9.0, 18.0, 273, 64, 35, 38.0,
    ("m4 pro", "m4pro", "apple m4 pro"),
)
_APPLE_M4_MAX = _apple(
    "Apple M4 Max", "M4", 18.0, 36.0, 546, 128, 55, 38.0,
    ("m4 max", "m4max", "apple m4 max"),
)
_APPLE_M4_ULTRA = _apple(
    "Apple M4 Ultra", "M4", 36.0, 72.0, 800, 192, 110, 76.0,
    ("m4 ultra", "m4ultra", "apple m4 ultra"),
)

# M5 family (announced Q1 2026 — specs from Apple product pages)
_APPLE_M5 = _apple(
    "Apple M5", "M5", 5.5, 11.0, 140, 32, 25, 50.0,
    ("m5", "apple m5"),
)
_APPLE_M5_PRO = _apple(
    "Apple M5 Pro", "M5", 11.0, 22.0, 300, 64, 40, 50.0,
    ("m5 pro", "m5pro", "apple m5 pro"),
)
_APPLE_M5_MAX = _apple(
    "Apple M5 Max", "M5", 22.0, 44.0, 600, 128, 60, 50.0,
    ("m5 max", "m5max", "apple m5 max"),
)
_APPLE_M5_ULTRA = _apple(
    "Apple M5 Ultra", "M5", 44.0, 88.0, 800, 192, 120, 100.0,
    ("m5 ultra", "m5ultra", "apple m5 ultra"),
)

# Raspberry Pi 4
_RASPBERRY_PI4 = HWSpec(
    name="Raspberry Pi 4",
    vendor="broadcom",
    architecture="ARM Cortex-A72",
    peak_flops={
        "float32": 48.0e9,
        "float16": 96.0e9,
    },
    peak_mem_bw=25.6e9,      # ≈ 25.6 GB/s LPDDR4
    memory_capacity_gb=8,
    memory_type="LPDDR4",
    tdp_watts=10,
    aliases=("pi4", "raspberry pi 4", "raspberry_pi4", "raspberrypi4", "bcm2711"),
)

# Raspberry Pi 5
_RASPBERRY_PI5 = HWSpec(
    name="Raspberry Pi 5",
    vendor="broadcom",
    architecture="ARM Cortex-A76",
    peak_flops={
        "float32": 38.4e9,
        "float16": 76.8e9,
    },
    peak_mem_bw=51.2e9,      # ≈ 51.2 GB/s LPDDR5
    memory_capacity_gb=16,
    memory_type="LPDDR5",
    tdp_watts=12,
    aliases=("pi5", "raspberry pi 5", "raspberry_pi5", "raspberrypi5", "bcm2712"),
)

# Arduino Nicla Vision
_ARDUINO_NICLA = HWSpec(
    name="Arduino Nicla Vision",
    vendor="arduino",
    architecture="ARM Cortex-M7",
    peak_flops={
        "float32": 1.92e9,
        "float16": 3.84e9,
    },
    peak_mem_bw=15.36e9,     # ≈ 15.36 GB/s dual-port SRAM
    memory_capacity_gb=0.002,
    memory_type="SRAM",
    tdp_watts=1.5,
    aliases=("nicla", "arduino nicla", "arduino_nicla", "nicla_vision", "stm32h747"),
)

# ---------------------------------------------------------------------------
# Registry — maps key → HWSpec
# ---------------------------------------------------------------------------

HW_DB: Dict[str, HWSpec] = {
    # NVIDIA data-centre
    "v100_sxm": _NVIDIA_V100_SXM,
    "a100_40gb": _NVIDIA_A100_40,
    "a100_80gb": _NVIDIA_A100_80,
    "h100_sxm": _NVIDIA_H100_SXM,
    "h100_pcie": _NVIDIA_H100_PCIE,
    "h200_sxm": _NVIDIA_H200_SXM,
    # NVIDIA consumer
    "rtx_3090": _NVIDIA_RTX_3090,
    "rtx_4090": _NVIDIA_RTX_4090,
    "rtx_5090": _NVIDIA_RTX_5090,
    # AMD
    "mi300x": _AMD_MI300X,
    # Intel
    "gaudi2": _INTEL_GAUDI2,
    "gaudi3": _INTEL_GAUDI3,
    # Apple M2
    "m2": _APPLE_M2,
    "m2_pro": _APPLE_M2_PRO,
    "m2_max": _APPLE_M2_MAX,
    "m2_ultra": _APPLE_M2_ULTRA,
    # Apple M3
    "m3": _APPLE_M3,
    "m3_pro": _APPLE_M3_PRO,
    "m3_max": _APPLE_M3_MAX,
    "m3_ultra": _APPLE_M3_ULTRA,
    # Apple M4
    "m4": _APPLE_M4,
    "m4_pro": _APPLE_M4_PRO,
    "m4_max": _APPLE_M4_MAX,
    "m4_ultra": _APPLE_M4_ULTRA,
    # Apple M5
    "m5": _APPLE_M5,
    "m5_pro": _APPLE_M5_PRO,
    "m5_max": _APPLE_M5_MAX,
    "m5_ultra": _APPLE_M5_ULTRA,
    "raspberry_pi4": _RASPBERRY_PI4,
    "raspberry_pi5": _RASPBERRY_PI5,
    "arduino_nicla": _ARDUINO_NICLA,
}

# Build a flat alias lookup for the fuzzy matcher in hw_resolver
_ALIAS_MAP: Dict[str, str] = {}
for _key, _spec in HW_DB.items():
    _ALIAS_MAP[_key] = _key
    _ALIAS_MAP[_spec.name.lower()] = _key
    for _alias in _spec.aliases:
        _ALIAS_MAP[_alias.lower()] = _key


def lookup(query: str) -> "HWSpec | None":
    """Return an HWSpec by exact key or alias, case-insensitive. Returns None if not found."""
    return HW_DB.get(_ALIAS_MAP.get(query.lower()))
