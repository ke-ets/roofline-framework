"""HuggingFace model parser.

Loads a model from the HuggingFace Hub (by name) or a local directory
via ``AutoModel.from_pretrained``, then delegates to ``PyTorchParser``
for layer-level analysis.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from roofline.core.layer_info import LayerInfo
from roofline.parsers.base import BaseParser


class HuggingFaceParser(BaseParser):
    """Parse a HuggingFace model (Hub name or local path) into ``List[LayerInfo]``."""

    def parse(
        self,
        model,
        input_shapes: Optional[List[Tuple]] = None,
        dtype: str = "float32",
    ) -> List[LayerInfo]:
        try:
            import transformers
        except ImportError as e:
            raise ImportError(
                "transformers is required for HuggingFace parsing. "
                "Install with: pip install transformers"
            ) from e

        import torch

        if isinstance(model, str):
            # Load from Hub or local directory
            model_obj = self._load_hf_model(model, dtype)
            model_path = model
        else:
            # Already an nn.Module (possibly from transformers)
            model_obj = model
            model_path = type(model).__name__

        # Infer input shapes from model config if not provided
        if input_shapes is None:
            input_shapes = self._infer_input_shapes(model_obj)

        # Delegate to PyTorchParser
        from roofline.parsers.pytorch_parser import PyTorchParser
        layers = PyTorchParser().parse(model_obj, input_shapes=input_shapes, dtype=dtype)

        # Tag layers with source_framework = "huggingface"
        for layer in layers:
            layer.source_framework = "huggingface"

        return layers

    # ------------------------------------------------------------------

    def _load_hf_model(self, model_name_or_path: str, dtype: str):
        """Load a HuggingFace model in eval mode, respecting dtype."""
        from transformers import AutoModel
        import torch

        torch_dtype = _dtype_to_torch(dtype)
        print(f"[roofline] Loading HuggingFace model: {model_name_or_path}")
        model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        model.eval()
        return model

    def _infer_input_shapes(self, model) -> Optional[List[Tuple]]:
        """Try to infer default input shapes from model config."""
        try:
            cfg = model.config
            # Sequence models: batch=1, seq_len=128
            if hasattr(cfg, "max_position_embeddings"):
                seq_len = min(128, cfg.max_position_embeddings)
                return [(1, seq_len)]
            # Vision models: batch=1, C=3, H=W=224
            if hasattr(cfg, "image_size"):
                img_size = cfg.image_size
                if isinstance(img_size, int):
                    return [(1, 3, img_size, img_size)]
                if isinstance(img_size, (list, tuple)):
                    return [(1, 3, *img_size)]
        except Exception:
            pass
        return None


def _dtype_to_torch(dtype: str):
    import torch
    _map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return _map.get(dtype.lower(), torch.float32)
