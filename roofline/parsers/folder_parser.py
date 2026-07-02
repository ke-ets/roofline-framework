"""FolderParser — loads a model from a directory.

Auto-detects which loading strategy to use:

Strategy A — HuggingFace local save (``config.json`` present):
    Delegates to ``HuggingFaceParser``.

Strategy B — Manual layer files (no ``config.json``):
    Scans for ``.pt``, ``.pth``, ``.safetensors``, ``.onnx``, ``.h5``,
    ``.keras`` files, sorts them lexicographically, loads each one
    individually, and concatenates the resulting ``LayerInfo`` lists.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Tuple

from roofline.core.layer_info import LayerInfo
from roofline.parsers.base import BaseParser

_SUPPORTED_EXTENSIONS = {".pt", ".pth", ".safetensors", ".onnx", ".h5", ".keras"}


class FolderParser(BaseParser):
    """Parse a directory containing model files into ``List[LayerInfo]``."""

    def parse(
        self,
        model,
        input_shapes: Optional[List[Tuple]] = None,
        dtype: str = "float32",
    ) -> List[LayerInfo]:
        folder = Path(model)
        if not folder.is_dir():
            raise ValueError(f"FolderParser expected a directory, got: {model}")

        # --- Strategy A: HuggingFace local save ---
        if (folder / "config.json").exists():
            print(f"[roofline] Detected HuggingFace local save in '{folder}'")
            from roofline.parsers.huggingface_parser import HuggingFaceParser
            return HuggingFaceParser().parse(str(folder), input_shapes=input_shapes, dtype=dtype)

        # --- Strategy B: Manual layer files ---
        files = sorted(
            [f for f in folder.iterdir() if f.suffix.lower() in _SUPPORTED_EXTENSIONS],
            key=lambda f: f.name,
        )

        if not files:
            raise ValueError(
                f"No supported model files found in '{folder}'. "
                f"Supported extensions: {sorted(_SUPPORTED_EXTENSIONS)}"
            )

        print(f"[roofline] Found {len(files)} layer file(s) in '{folder}' — loading sequentially.")

        all_layers: List[LayerInfo] = []
        for i, filepath in enumerate(files):
            print(f"[roofline]   [{i+1}/{len(files)}] {filepath.name}")
            layers = self._load_file(filepath, input_shapes=input_shapes, dtype=dtype)
            all_layers.extend(layers)

        print(f"[roofline] Total layers extracted: {len(all_layers)}")
        return all_layers

    def _load_file(
        self,
        filepath: Path,
        input_shapes: Optional[List[Tuple]],
        dtype: str,
    ) -> List[LayerInfo]:
        ext = filepath.suffix.lower()

        if ext in (".pt", ".pth"):
            return self._load_pytorch_file(filepath, input_shapes, dtype)
        elif ext == ".safetensors":
            return self._load_safetensors_file(filepath, input_shapes, dtype)
        elif ext == ".onnx":
            from roofline.parsers.onnx_parser import ONNXParser
            return ONNXParser().parse(str(filepath), input_shapes=input_shapes, dtype=dtype)
        elif ext in (".h5", ".keras"):
            from roofline.parsers.tensorflow_parser import TensorFlowParser
            return TensorFlowParser().parse(str(filepath), input_shapes=input_shapes, dtype=dtype)
        else:
            warnings.warn(f"Unsupported extension '{ext}' for file '{filepath.name}', skipping.")
            return []

    def _load_pytorch_file(self, filepath: Path, input_shapes, dtype) -> List[LayerInfo]:
        try:
            import torch
            obj = torch.load(str(filepath), map_location="cpu", weights_only=False)
        except Exception as e:
            try:
                # weights_only=True is the new default in PyTorch 2.x for raw tensors
                import torch
                obj = torch.load(str(filepath), map_location="cpu", weights_only=True)
            except Exception:
                warnings.warn(f"Could not load '{filepath.name}': {e}", stacklevel=2)
                return []

        # Full nn.Module
        if hasattr(obj, "forward"):
            from roofline.parsers.pytorch_parser import PyTorchParser
            return PyTorchParser().parse(obj, input_shapes=input_shapes, dtype=dtype)

        # State dict (OrderedDict / dict of tensors)
        if isinstance(obj, dict):
            return self._state_dict_to_layer_info(obj, filepath.stem, dtype)

        # Raw tensor — single weight/bias file (e.g. "features_0_weight.pt")
        import torch
        if isinstance(obj, torch.Tensor):
            stem = filepath.stem  # e.g. "features_0_weight"
            return self._state_dict_to_layer_info({stem: obj}, stem, dtype)

        warnings.warn(
            f"'{filepath.name}' contains an unrecognised object ({type(obj).__name__}). "
            f"Skipping.",
            stacklevel=2,
        )
        return []

    def _load_safetensors_file(self, filepath: Path, input_shapes, dtype) -> List[LayerInfo]:
        try:
            from safetensors import safe_open
        except ImportError as e:
            raise ImportError(
                "safetensors is required for .safetensors files. "
                "Install with: pip install safetensors"
            ) from e

        tensors: dict = {}
        with safe_open(str(filepath), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

        return self._state_dict_to_layer_info(tensors, filepath.stem, dtype)

    @staticmethod
    def _state_dict_to_layer_info(state_dict: dict, prefix: str, dtype: str) -> List[LayerInfo]:
        """Convert a flat state_dict (weight tensors) into synthetic LayerInfo records.

        Handles two naming conventions:
          dot-separated  : ``layer1.0.weight``, ``layer1.0.bias``
          underscore-sep : ``features_0_weight``, ``features_0_bias``  (one file = one tensor)
        """
        from collections import defaultdict

        groups: dict = defaultdict(list)
        for key, tensor in state_dict.items():
            # Underscore-separated convention: strip trailing _weight / _bias
            clean = key
            for suffix in ("_weight", "_bias", ".weight", ".bias"):
                if key.endswith(suffix):
                    clean = key[: -len(suffix)]
                    break
            groups[clean].append((key, tensor))

        layers = []
        for group_name, tensors in groups.items():
            num_params = sum(t.numel() for _, t in tensors)
            # Find the weight tensor (largest or name-matched)
            weight_tensor = None
            for key, tensor in tensors:
                if "weight" in key:
                    weight_tensor = tensor
                    break
            if weight_tensor is None and tensors:
                weight_tensor = tensors[0][1]

            if weight_tensor is not None:
                layer_type = _infer_type_from_shape(weight_tensor.shape)
                attrs = _attrs_from_shape(weight_tensor.shape)
                # Add bias flag if a bias tensor exists
                attrs["bias"] = any("bias" in k for k, _ in tensors)
                layers.append(
                    LayerInfo(
                        name=f"{prefix}/{group_name}" if prefix != group_name else group_name,
                        layer_type=layer_type,
                        input_shapes=[],
                        output_shapes=[],
                        num_params=num_params,
                        dtype=dtype,
                        attrs=attrs,
                        source_framework="pytorch",
                    )
                )

        return layers


def _infer_type_from_shape(shape) -> str:
    ndim = len(shape)
    if ndim == 2:
        return "Linear"
    elif ndim == 3:
        return "Conv1d"
    elif ndim == 4:
        return "Conv2d"
    elif ndim == 5:
        return "Conv3d"
    return "Unknown"


def _attrs_from_shape(shape) -> dict:
    ndim = len(shape)
    if ndim == 2:
        return {"out_features": shape[0], "in_features": shape[1]}
    elif ndim >= 3:
        return {
            "out_channels": shape[0],
            "in_channels": shape[1],
            "kernel_size": shape[2:],
        }
    return {}
