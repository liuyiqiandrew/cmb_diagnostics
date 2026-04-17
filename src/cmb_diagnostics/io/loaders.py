"""Map loaders for Planck (HEALPix equatorial) and SO (CAR)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from cmb_diagnostics._types import Tracer

if TYPE_CHECKING:
    from cmb_diagnostics.config import InstrumentConfig


@runtime_checkable
class MapLoader(Protocol):
    def load(self, tracer: Tracer) -> np.ndarray:
        """Return the HEALPix map for ``tracer`` at the configured nside, in muK."""
        ...


class PlanckHealpixLoader:
    def __init__(self, cfg: InstrumentConfig, nside: int) -> None:
        self.cfg = cfg
        self.nside = nside

    def load(self, tracer: Tracer) -> np.ndarray:
        """Phase 3: read FITS via healpy.read_map, ud_grade to nside, multiply by
        cfg.unit_scale. Source: V1 ``PSContainer.init_planck_f2`` map read.
        """
        raise NotImplementedError(
            "Phase 3: port from cmb_diagnoistics/PSContainer.py::PSContainer.init_planck_f2 "
            "(healpy.read_map + ud_grade + unit_scale)."
        )


class SOCarLoader:
    def __init__(self, cfg: InstrumentConfig, nside: int) -> None:
        self.cfg = cfg
        self.nside = nside

    def load(self, tracer: Tracer) -> np.ndarray:
        """Phase 3: read CAR via pixell.enmap.read_map, reproject to HEALPix,
        ud_grade, unit_scale. Source: ``cmb_diagnoistics/diag_utils.py::read_carr2healpix``.
        """
        raise NotImplementedError(
            "Phase 3: port from cmb_diagnoistics/diag_utils.py::read_carr2healpix "
            "(pixell + reproject.map2healpix + ud_grade + unit_scale)."
        )


def get_loader(cfg: InstrumentConfig, nside: int) -> MapLoader:
    """Factory: dispatch on ``cfg.pixelization``.

    Phase 3: returns ``PlanckHealpixLoader`` or ``SOCarLoader``.
    """
    raise NotImplementedError(
        "Phase 3: dispatch on cfg.pixelization -> PlanckHealpixLoader|SOCarLoader."
    )
