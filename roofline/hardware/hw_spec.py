"""HWSpec dataclass — describes a hardware target's peak capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class HWSpec:
    """Hardware specification used for roofline model computation.

    All bandwidth and compute values are stored in SI base units
    (bytes/second and FLOPs/second) to keep arithmetic unambiguous.
    """

    name: str
    """Human-readable hardware name, e.g. ``"NVIDIA H100 SXM 80GB"``."""

    peak_flops: Dict[str, float]
    """Peak throughput in FLOPs/second keyed by precision.

    Required keys (if the hardware supports them):
    ``"float32"``, ``"float16"``, ``"bfloat16"``, ``"int8"``, ``"int4"``

    Example (H100 SXM)::

        {"float32": 67e12, "float16": 989e12, "bfloat16": 989e12, "int8": 1979e12}
    """

    peak_mem_bw: float
    """Peak memory bandwidth in bytes/second.
    e.g. H100 SXM HBM3 = 3.35e12 (3.35 TB/s).
    """

    memory_capacity_gb: float
    """Total memory capacity in gigabytes."""

    memory_type: str
    """Memory technology string, e.g. ``"HBM2e"``, ``"HBM3"``,
    ``"GDDR6X"``, ``"GDDR7"``, ``"LPDDR5X"``."""

    tdp_watts: float
    """Thermal Design Power in watts."""

    unified_memory: bool = False
    """True for Apple Silicon — CPU and GPU share the same memory pool.
    On unified-memory platforms there is no separate VRAM; the same
    ``peak_mem_bw`` applies to all accesses.
    """

    ane_tops: float = 0.0
    """Apple Neural Engine throughput in TOPS (INT8).
    0.0 for non-Apple hardware.
    """

    vendor: str = ""
    """Vendor shortname: ``"nvidia"``, ``"amd"``, ``"intel"``, ``"apple"``."""

    architecture: str = ""
    """Microarchitecture name, e.g. ``"Hopper"``, ``"Ada Lovelace"``,
    ``"CDNA3"``, ``"Gaudi2"``, ``"M4"``."""

    # Aliases used by the fuzzy matcher in hw_detector / hw_resolver
    aliases: tuple = field(default_factory=tuple)

    def ridge_point(self, dtype: str = "float16") -> float:
        """Return the ridge point (FLOPs/Byte) for the given dtype.

        The ridge point is the arithmetic intensity at which compute and
        memory bandwidth are equally constraining:

            ridge = peak_flops[dtype] / peak_mem_bw
        """
        dtype = dtype.lower()
        flops = self._get_peak_flops(dtype)
        if self.peak_mem_bw == 0:
            return float("inf")
        return flops / self.peak_mem_bw

    def attainable_performance(self, arithmetic_intensity: float, dtype: str = "float16") -> float:
        """Return attainable performance (FLOPs/s) for a given arithmetic intensity.

            attainable = min(AI * peak_mem_bw, peak_flops[dtype])
        """
        dtype = dtype.lower()
        flops = self._get_peak_flops(dtype)
        return min(arithmetic_intensity * self.peak_mem_bw, flops)

    def _get_peak_flops(self, dtype: str) -> float:
        """Return peak FLOPs for dtype, falling back to nearest available."""
        dtype = dtype.lower()
        if dtype in self.peak_flops:
            return self.peak_flops[dtype]
        # Fallback chain
        fallback_order = ["float16", "bfloat16", "float32", "int8"]
        for fb in fallback_order:
            if fb in self.peak_flops:
                return self.peak_flops[fb]
        return max(self.peak_flops.values()) if self.peak_flops else 0.0

    def __str__(self) -> str:
        bw_tbs = self.peak_mem_bw / 1e12
        flops_list = ", ".join(
            f"{k}: {v/1e12:.0f}T" for k, v in self.peak_flops.items()
        )
        return (
            f"{self.name} | BW: {bw_tbs:.2f} TB/s | "
            f"FLOPs: {{{flops_list}}} | "
            f"Mem: {self.memory_capacity_gb}GB {self.memory_type} | "
            f"TDP: {self.tdp_watts}W"
        )
