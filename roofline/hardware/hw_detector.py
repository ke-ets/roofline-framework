"""Local hardware auto-detection.

Tries to identify the GPU (or CPU/ANE for Apple Silicon) that this
machine is currently running on, then matches the detected name against
the built-in ``HW_DB``.

Resolution order:
  1. NVIDIA GPU via ``pynvml`` (no subprocess)
  2. AMD GPU via ``rocm-smi`` subprocess
  3. Apple Silicon via ``sysctl`` / ``system_profiler`` on macOS arm64
  4. CPU fallback via ``platform``

Returns a matched ``HWSpec`` from the built-in database, or raises
``HWDetectionError`` if the hardware could not be matched and
instructs the caller to use ``fetch_hw()`` to retrieve specs from the web.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Optional

from roofline.hardware.hw_spec import HWSpec


class HWDetectionError(Exception):
    """Raised when local hardware cannot be identified or matched."""


def detect_hw(quiet: bool = False) -> HWSpec:
    """Detect the local GPU/accelerator and return its ``HWSpec``.

    Tries NVIDIA → AMD → Apple Silicon → CPU (fallback) in order.

    Parameters
    ----------
    quiet:
        Suppress informational print output.

    Returns
    -------
    HWSpec
        Matched entry from ``HW_DB``.

    Raises
    ------
    HWDetectionError
        If the detected hardware is not in the built-in database.
        The exception message includes the detected name so the caller
        can pass it to ``fetch_hw()``.
    """
    name, raw_name = None, None

    # 1. NVIDIA via pynvml
    name, raw_name = _try_nvidia()

    # 2. AMD via rocm-smi
    if name is None:
        name, raw_name = _try_amd()

    # 3. Apple Silicon
    if name is None:
        name, raw_name = _try_apple()

    # 4. CPU fallback
    if name is None:
        raw_name = platform.processor() or platform.machine()
        name = raw_name

    if not quiet:
        print(f"[roofline] Detected hardware: {raw_name}")

    # Match against the built-in database
    from roofline.hardware.hw_database import lookup
    spec = lookup(name)
    if spec is None:
        raise HWDetectionError(
            f"Detected hardware '{raw_name}' is not in the built-in database. "
            f"Run fetch_hw('{raw_name}', fetch_from_web=True) to fetch specs from the web, "
            f"or choose a key from HW_DB: {_hw_db_keys()}"
        )
    if not quiet:
        print(f"[roofline] Matched to: {spec.name}")
    return spec


# ---------------------------------------------------------------------------
# Backend probes
# ---------------------------------------------------------------------------

def _try_nvidia():
    """Probe NVIDIA GPU via pynvml. Returns (matched_key_candidate, raw_name)."""
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        raw_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(raw_name, bytes):
            raw_name = raw_name.decode()
        pynvml.nvmlShutdown()
        return _normalise(raw_name), raw_name
    except Exception:
        return None, None


def _try_amd():
    """Probe AMD GPU via rocm-smi subprocess."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None, None
        import json
        data = json.loads(result.stdout)
        # rocm-smi JSON: {"card0": {"Card series": "...", "Card model": "..."}, ...}
        for card_data in data.values():
            raw_name = card_data.get("Card series") or card_data.get("Card model")
            if raw_name:
                return _normalise(raw_name), raw_name
    except Exception:
        pass
    return None, None


def _try_apple():
    """Probe Apple Silicon chip name via sysctl / system_profiler."""
    if sys.platform != "darwin" or platform.machine() not in ("arm64", "aarch64"):
        return None, None
    try:
        # machdep.cpu.brand_string gives "Apple M4 Pro" etc.
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        raw_name = result.stdout.strip()
        if raw_name and "Apple" in raw_name:
            return _normalise(raw_name), raw_name
    except Exception:
        pass
    # Fallback: system_profiler
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            if "Chip" in line or "Processor Name" in line:
                raw_name = line.split(":")[-1].strip()
                if raw_name:
                    return _normalise(raw_name), raw_name
    except Exception:
        pass
    return None, None


def _normalise(name: str) -> str:
    """Lowercase and strip for alias matching."""
    return name.lower().strip()


def _hw_db_keys():
    from roofline.hardware.hw_database import HW_DB
    return list(HW_DB.keys())
