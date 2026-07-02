"""ZipHandler — transparent pre-processor for .zip model archives.

Extracts the archive to a temporary directory, delegates to
``FolderParser``, then cleans up on completion or error.

Both flat zips (files at root) and nested zips (single top-level
subfolder) are handled transparently.
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

from roofline.core.layer_info import LayerInfo
from roofline.parsers.base import BaseParser


class ZipHandler(BaseParser):
    """Parse a ``.zip`` model archive into ``List[LayerInfo]``."""

    def parse(
        self,
        model,
        input_shapes: Optional[List[Tuple]] = None,
        dtype: str = "float32",
    ) -> List[LayerInfo]:
        zip_path = Path(model)
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")
        if not zipfile.is_zipfile(str(zip_path)):
            raise ValueError(f"Not a valid zip file: {zip_path}")

        tmpdir = tempfile.mkdtemp(prefix="roofline_zip_")
        try:
            print(f"[roofline] Extracting '{zip_path.name}' to temporary directory…")
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                zf.extractall(tmpdir)

            # If zip has a single top-level subfolder, descend into it
            extract_dir = _resolve_extract_root(tmpdir)
            print(f"[roofline] Extraction root: {extract_dir}")

            from roofline.parsers.folder_parser import FolderParser
            return FolderParser().parse(
                extract_dir,
                input_shapes=input_shapes,
                dtype=dtype,
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            print(f"[roofline] Cleaned up temporary directory.")


def _resolve_extract_root(tmpdir: str) -> str:
    """If the zip extracted a single top-level folder, return that; else return tmpdir."""
    items = list(Path(tmpdir).iterdir())
    if len(items) == 1 and items[0].is_dir():
        return str(items[0])
    return tmpdir
