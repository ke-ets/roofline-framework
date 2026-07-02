"""Abstract base class for all model parsers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from roofline.core.layer_info import LayerInfo


class BaseParser(ABC):
    """All parsers convert a model representation into ``List[LayerInfo]``."""

    @abstractmethod
    def parse(
        self,
        model,
        input_shapes: Optional[List[Tuple]] = None,
        dtype: str = "float32",
    ) -> List[LayerInfo]:
        """Parse the model and return one ``LayerInfo`` per layer.

        Parameters
        ----------
        model:
            The model object or path understood by this parser.
        input_shapes:
            List of (batch, ...) tuples for each model input.
        dtype:
            Precision string used to populate ``LayerInfo.dtype``.
        """

    @staticmethod
    def _prod(shape: Tuple) -> int:
        result = 1
        for d in shape:
            if d is not None and d > 0:
                result *= d
        return result
