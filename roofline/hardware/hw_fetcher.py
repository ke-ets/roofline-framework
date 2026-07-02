"""Fetch hardware specs from TechPowerUp GPU Database.

Web fetching is **always opt-in** — no network calls are made unless
the caller explicitly passes ``fetch_from_web=True`` (Python API) or
confirms the CLI prompt.

Fetched specs are cached in ``~/.roofline/hw_cache.json`` so subsequent
calls do not require network access.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from roofline.hardware.hw_spec import HWSpec

_CACHE_PATH = Path.home() / ".roofline" / "hw_cache.json"
_TECHPOWERUP_SEARCH = "https://www.techpowerup.com/gpu-specs/?mfgr={vendor}&q={query}&ajaxsrch=1"
_TECHPOWERUP_BASE = "https://www.techpowerup.com"

# Regex helpers for parsing TechPowerUp spec pages
_RE_NUMBER = re.compile(r"[\d,.]+")


class HWFetchError(Exception):
    """Raised when specs cannot be fetched or parsed."""


def fetch_hw(
    name: str,
    fetch_from_web: bool = False,
    interactive: bool = False,
) -> HWSpec:
    """Return an ``HWSpec`` for the named hardware.

    Resolution order:
    1. Built-in ``HW_DB`` (instant, no network)
    2. Local cache ``~/.roofline/hw_cache.json``
    3. Web fetch from TechPowerUp (requires opt-in)

    Parameters
    ----------
    name:
        Hardware name or alias, e.g. ``"RTX 5090"``, ``"H100 SXM"``.
    fetch_from_web:
        Set to ``True`` to allow network requests.  When ``False`` and
        the spec is not cached, raises ``HWFetchError``.
    interactive:
        When ``True`` and ``fetch_from_web`` is ``False``, prompt the
        user interactively for approval before fetching.
    """
    # 1. Built-in DB
    from roofline.hardware.hw_database import lookup
    spec = lookup(name)
    if spec is not None:
        return spec

    # 2. Local cache
    spec = _load_from_cache(name)
    if spec is not None:
        print(f"[roofline] Loaded '{name}' from local cache.")
        return spec

    # 3. Web fetch — requires approval
    if not fetch_from_web:
        if interactive:
            answer = input(f"Fetch specs for '{name}' from TechPowerUp? [y/N]: ").strip().lower()
            if answer not in ("y", "yes"):
                raise HWFetchError(
                    f"User declined web fetch for '{name}'. "
                    f"Pass fetch_from_web=True to allow it programmatically."
                )
            fetch_from_web = True
        else:
            raise HWFetchError(
                f"'{name}' not found in built-in DB or local cache. "
                f"Pass fetch_from_web=True to fetch from TechPowerUp."
            )

    print(f"[roofline] Fetching specs for '{name}' from TechPowerUp GPU Database…")
    spec = _fetch_from_techpowerup(name)
    _save_to_cache(name, spec)
    print(f"[roofline] Cached to {_CACHE_PATH}")
    return spec


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "_", name.lower()).strip("_")


def _load_from_cache(name: str) -> Optional[HWSpec]:
    if not _CACHE_PATH.exists():
        return None
    try:
        with open(_CACHE_PATH) as f:
            data: Dict = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    key = _cache_key(name)
    if key not in data:
        return None
    entry = data[key]
    return HWSpec(
        name=entry["name"],
        vendor=entry.get("vendor", ""),
        architecture=entry.get("architecture", ""),
        peak_flops={k: float(v) for k, v in entry["peak_flops"].items()},
        peak_mem_bw=float(entry["peak_mem_bw"]),
        memory_capacity_gb=float(entry["memory_capacity_gb"]),
        memory_type=entry.get("memory_type", ""),
        tdp_watts=float(entry.get("tdp_watts", 0)),
        unified_memory=entry.get("unified_memory", False),
        ane_tops=float(entry.get("ane_tops", 0)),
    )


def _save_to_cache(name: str, spec: HWSpec) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data: Dict = {}
    if _CACHE_PATH.exists():
        try:
            with open(_CACHE_PATH) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}

    key = _cache_key(name)
    data[key] = {
        "name": spec.name,
        "vendor": spec.vendor,
        "architecture": spec.architecture,
        "peak_flops": spec.peak_flops,
        "peak_mem_bw": spec.peak_mem_bw,
        "memory_capacity_gb": spec.memory_capacity_gb,
        "memory_type": spec.memory_type,
        "tdp_watts": spec.tdp_watts,
        "unified_memory": spec.unified_memory,
        "ane_tops": spec.ane_tops,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(_CACHE_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# TechPowerUp scraper
# ---------------------------------------------------------------------------

def _fetch_from_techpowerup(name: str) -> HWSpec:
    """Scrape TechPowerUp GPU specs for the given GPU name."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise HWFetchError(
            "requests and beautifulsoup4 are required for web fetching. "
            "Install with: pip install requests beautifulsoup4"
        ) from e

    # Search TechPowerUp
    search_url = f"https://www.techpowerup.com/gpu-specs/?q={requests.utils.quote(name)}"
    headers = {"User-Agent": "roofline-framework/0.1 (GPU spec lookup)"}

    try:
        resp = requests.get(search_url, headers=headers, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise HWFetchError(f"Network error fetching specs for '{name}': {e}") from e

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find first GPU result link
    spec_url = _find_spec_url(soup, name)
    if spec_url is None:
        raise HWFetchError(
            f"No TechPowerUp result found for '{name}'. "
            f"Try a different name or add specs manually via HWSpec(...)."
        )

    # Fetch the spec detail page
    try:
        detail_resp = requests.get(spec_url, headers=headers, timeout=15)
        detail_resp.raise_for_status()
    except requests.RequestException as e:
        raise HWFetchError(f"Could not fetch spec page {spec_url}: {e}") from e

    detail_soup = BeautifulSoup(detail_resp.text, "html.parser")
    return _parse_spec_page(detail_soup, spec_url)


def _find_spec_url(soup, query: str) -> Optional[str]:
    """Extract the first GPU spec page URL from a TechPowerUp search results page."""
    # TechPowerUp search results: table rows with links like /gpu-specs/...html
    for a in soup.select("table a[href*='/gpu-specs/']"):
        href = a.get("href", "")
        if href.endswith(".html"):
            return _TECHPOWERUP_BASE + href if href.startswith("/") else href
    return None


def _parse_spec_page(soup, source_url: str) -> HWSpec:
    """Parse a TechPowerUp GPU detail page into an HWSpec."""
    def _get_spec(label: str) -> str:
        """Find a spec row by label text and return its value."""
        for dt in soup.select("dt, th"):
            if label.lower() in dt.get_text(strip=True).lower():
                dd = dt.find_next_sibling("dd") or dt.find_next_sibling("td")
                if dd:
                    return dd.get_text(strip=True)
        return ""

    gpu_name = ""
    h1 = soup.find("h1")
    if h1:
        gpu_name = h1.get_text(strip=True)

    # Memory bandwidth: "X GB/s" or "X TB/s"
    bw_raw = _get_spec("Memory Bandwidth")
    peak_mem_bw = _parse_bandwidth(bw_raw)

    # Memory size: "X GB"
    mem_raw = _get_spec("Memory Size")
    memory_gb = _parse_gb(mem_raw)

    # Memory type
    mem_type = _get_spec("Memory Type") or ""

    # TDP: "X W"
    tdp_raw = _get_spec("TDP") or _get_spec("Board TDP")
    tdp = _parse_watts(tdp_raw)

    # FP32 shading units / Pixel fillrate → approximate TFLOPS
    # TechPowerUp lists "Pixel Rate", "Texture Rate", sometimes "FP32 (float)" directly
    fp32_raw = _get_spec("FP32 (float)") or _get_spec("Single-precision")
    fp32_tflops = _parse_tflops(fp32_raw)

    # Build conservative FLOPs dict (FP16 = 2× FP32 on most modern GPUs)
    peak_flops: Dict[str, float] = {}
    if fp32_tflops > 0:
        peak_flops["float32"] = fp32_tflops * 1e12
        peak_flops["float16"] = fp32_tflops * 2 * 1e12
        peak_flops["bfloat16"] = fp32_tflops * 2 * 1e12
        peak_flops["int8"] = fp32_tflops * 4 * 1e12

    vendor = ""
    if "nvidia" in gpu_name.lower() or "geforce" in gpu_name.lower() or "quadro" in gpu_name.lower():
        vendor = "nvidia"
    elif "amd" in gpu_name.lower() or "radeon" in gpu_name.lower():
        vendor = "amd"
    elif "intel" in gpu_name.lower() or "arc" in gpu_name.lower():
        vendor = "intel"
    elif "apple" in gpu_name.lower():
        vendor = "apple"

    return HWSpec(
        name=gpu_name or "Unknown GPU",
        vendor=vendor,
        peak_flops=peak_flops if peak_flops else {"float32": 0.0},
        peak_mem_bw=peak_mem_bw,
        memory_capacity_gb=memory_gb,
        memory_type=mem_type,
        tdp_watts=tdp,
    )


# ---------------------------------------------------------------------------
# Value parsers
# ---------------------------------------------------------------------------

def _parse_bandwidth(raw: str) -> float:
    """Parse strings like '3,350.0 GB/s' or '3.35 TB/s' → bytes/sec."""
    raw = raw.replace(",", "")
    nums = _RE_NUMBER.findall(raw)
    if not nums:
        return 0.0
    val = float(nums[0])
    if "TB" in raw.upper():
        return val * 1e12
    return val * 1e9  # GB/s default


def _parse_gb(raw: str) -> float:
    raw = raw.replace(",", "")
    nums = _RE_NUMBER.findall(raw)
    return float(nums[0]) if nums else 0.0


def _parse_watts(raw: str) -> float:
    raw = raw.replace(",", "")
    nums = _RE_NUMBER.findall(raw)
    return float(nums[0]) if nums else 0.0


def _parse_tflops(raw: str) -> float:
    """Parse '67 TFLOPS', '67.2 TFLOPs', '989.4 GFLOPS' → value in TFLOPS."""
    raw = raw.replace(",", "")
    nums = _RE_NUMBER.findall(raw)
    if not nums:
        return 0.0
    val = float(nums[0])
    if "GFLOPS" in raw.upper() or "GFLOP/S" in raw.upper():
        return val / 1000.0
    return val  # assume TFLOPS
