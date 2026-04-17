"""Build NaMaster fields from raw maps + beams + mask."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from cmb_diagnostics.fields.container import FieldSet

if TYPE_CHECKING:
    from cmb_diagnostics.config import InstrumentConfig
    from cmb_diagnostics.io.masks import Mask


def build_spin0_field(
    tmap: np.ndarray,
    mask: Mask,
    beam_fwhm_arcmin: float,
    nside: int,
) -> Any:
    """Phase 3: build an ``nmt.NmtField`` with spin=0.

    Source: V1 ``PSContainer.init_planck_f0``.
    """
    raise NotImplementedError(
        "Phase 3: port from cmb_diagnoistics/PSContainer.py::PSContainer.init_planck_f0 "
        "(hp.gauss_beam + nmt.NmtField with [T] and the apodized mask)."
    )


def build_spin2_field(
    qmap: np.ndarray,
    umap: np.ndarray,
    mask: Mask,
    beam_fwhm_arcmin: float,
    nside: int,
    purify_e: bool = False,
    purify_b: bool = False,
) -> Any:
    """Phase 3: build an ``nmt.NmtField`` with spin=2.

    Source: V1 ``PSContainer.init_planck_f2`` / ``init_so_f2``.
    """
    raise NotImplementedError(
        "Phase 3: port from cmb_diagnoistics/PSContainer.py::PSContainer.init_planck_f2 "
        "(hp.gauss_beam + nmt.NmtField with [Q,U] + purify flags)."
    )


def build_fieldset(cfg: InstrumentConfig, mask: Mask, nside: int) -> FieldSet:
    """Phase 3: build a ``FieldSet`` by loading maps + constructing NmtField
    objects for each band in ``cfg.bands``.

    Source: composition of V1 ``PSContainer.init_planck_f0``/``_f2`` /
    ``init_so_f2``.
    """
    raise NotImplementedError(
        "Phase 3: read maps via loaders.get_loader(cfg), call build_spin0_field "
        "and build_spin2_field per band, return FieldSet(cfg.name)."
    )
