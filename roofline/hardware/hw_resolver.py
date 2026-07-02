"""HWResolver — single entry point for resolving any HW name to an HWSpec.

Resolution cascade (stops at first hit):
  1. Built-in ``HW_DB`` (exact key / alias match)
  2. Local cache ``~/.roofline/hw_cache.json``
  3. Local auto-detection via ``hw_detector.detect_hw()``
  4. Web fetch via ``hw_fetcher.fetch_hw()`` — only with user approval
"""

from __future__ import annotations

from typing import Optional

from roofline.hardware.hw_spec import HWSpec


class HWResolver:
    """Resolves a hardware name or None to an ``HWSpec``.

    Parameters
    ----------
    fetch_from_web:
        Allow web fetching without an interactive prompt.
    interactive:
        When True and ``fetch_from_web`` is False, ask the user before fetching.
    quiet:
        Suppress informational messages.
    """

    def __init__(
        self,
        fetch_from_web: bool = False,
        interactive: bool = False,
        quiet: bool = False,
    ):
        self.fetch_from_web = fetch_from_web
        self.interactive = interactive
        self.quiet = quiet

    def resolve(
        self,
        name: Optional[str] = None,
        detect_local: bool = False,
    ) -> HWSpec:
        """Return an ``HWSpec`` for the given name.

        Parameters
        ----------
        name:
            Hardware name string, or ``None`` when using local detection.
        detect_local:
            When ``True`` and ``name`` is None, try to detect the local GPU.
        """
        # 1. Built-in DB + alias lookup
        if name is not None:
            from roofline.hardware.hw_database import lookup
            spec = lookup(name)
            if spec is not None:
                return spec

        # 2. Local cache
        if name is not None:
            from roofline.hardware.hw_fetcher import _load_from_cache
            spec = _load_from_cache(name)
            if spec is not None:
                if not self.quiet:
                    print(f"[roofline] '{name}' loaded from local cache.")
                return spec

        # 3. Local detection
        if detect_local or name is None:
            try:
                from roofline.hardware.hw_detector import detect_hw
                spec = detect_hw(quiet=self.quiet)
                if spec is not None:
                    return spec
            except Exception:
                pass

        # 4. Web fetch
        if name is not None:
            from roofline.hardware.hw_fetcher import fetch_hw
            return fetch_hw(
                name,
                fetch_from_web=self.fetch_from_web,
                interactive=self.interactive,
            )

        raise ValueError(
            "Could not resolve hardware. Provide a name from HW_DB, "
            "set detect_local=True, or pass fetch_from_web=True with a name."
        )
