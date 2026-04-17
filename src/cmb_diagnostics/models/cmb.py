"""Binned CMB reference container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from cmb_diagnostics._types import Comp

if TYPE_CHECKING:
    from cmb_diagnostics.models.bandpowers import Bandpowers


@dataclass(frozen=True)
class CMBReference:
    bandpowers: Bandpowers
    cls_binned: dict[Comp, np.ndarray] = field(default_factory=dict)

    def get(self, comp: Comp) -> np.ndarray:
        if comp not in self.cls_binned:
            raise KeyError(f"CMBReference has no {comp!r} spectrum")
        return self.cls_binned[comp]

    def has(self, comp: Comp) -> bool:
        return comp in self.cls_binned
