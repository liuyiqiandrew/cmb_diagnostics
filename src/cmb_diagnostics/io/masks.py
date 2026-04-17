"""Mask dataclass + loaders.

Mask itself is a real frozen dataclass (pure data, no computation). The loader
and helper functions are stubs; Phase 3 ports them from diag_utils.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cmb_diagnostics.config import MaskConfig


@dataclass(frozen=True)
class Mask:
    hp_map: np.ndarray
    nside: int
    fsky_effective: float


def load_mask(cfg: MaskConfig) -> Mask:
    """Load a HEALPix mask according to ``cfg``.

    Phase 3: dispatch on ``cfg.kind`` (``"file"`` or ``"boxes"``), apodize per
    ``cfg.apodize``, apply ``cfg.threshold``, compute effective fsky. Source:
    V1 ``PSContainer.init_mask`` / ``init_mask_from_box`` + ``diag_utils.box2hpmask``.
    """
    raise NotImplementedError(
        "Phase 3: port from cmb_diagnoistics/PSContainer.py::PSContainer.init_mask "
        "and init_mask_from_box; see cmb_diagnoistics/diag_utils.py::box2hpmask."
    )


def box2hpmask(
    nside: int, boxes: list[list[list[float]]]
) -> np.ndarray:
    """Build a HEALPix boolean mask from a list of [[dec_min, ra_min], [dec_max, ra_max]] boxes in degrees.

    Phase 3: port from ``cmb_diagnoistics/diag_utils.py::box2hpmask``.
    """
    raise NotImplementedError(
        "Phase 3: port from cmb_diagnoistics/diag_utils.py::box2hpmask."
    )


def apodize_square_mask(mask: np.ndarray, nside: int) -> np.ndarray:
    """Apply healpy smoothing + NaMaster C2 apodization to a boolean mask.

    Phase 3: port from ``cmb_diagnoistics/diag_utils.py::apodize_square_mask``.
    """
    raise NotImplementedError(
        "Phase 3: port from cmb_diagnoistics/diag_utils.py::apodize_square_mask."
    )


def effective_fsky(mask: np.ndarray) -> float:
    """Return the apodization-weighted effective fsky: sum(w^2) / Npix.

    Phase 3: thin helper, new to refactor (replaces V1's sum(w)/Npix).
    """
    raise NotImplementedError("Phase 3: sum(mask**2) / mask.size.")
