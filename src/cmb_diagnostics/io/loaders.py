"""Map loaders for Planck (HEALPix equatorial) and SO (CAR).

Every loader returns a ``(3, npix)`` I/Q/U array in muK at the configured
``nside``. The ``fields.builder`` module slices T or [Q,U] as needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from cmb_diagnostics._types import Tracer

if TYPE_CHECKING:
    from cmb_diagnostics.config import InstrumentConfig


@runtime_checkable
class MapLoader(Protocol):
    def load(self, tracer: Tracer) -> np.ndarray:
        """Return a ``(3, npix)`` I/Q/U HEALPix map at the configured nside."""
        ...


def _format_path(template: str, tracer: Tracer) -> str:
    # Accept both '{freq}' and positional '{}' templates; V1 used the latter.
    freq_int = int(tracer.freq)
    try:
        return template.format(freq=freq_int)
    except KeyError:
        return template.format(freq_int)


class PlanckHealpixLoader:
    """Read Planck equatorial HEALPix T/Q/U FITS maps, ud_grade to nside, scale."""

    def __init__(self, cfg: InstrumentConfig, nside: int) -> None:
        self.cfg = cfg
        self.nside = nside

    def load(self, tracer: Tracer) -> np.ndarray:
        import healpy as hp

        path = _format_path(self.cfg.map_template, tracer)
        tqu = hp.read_map(path, field=[0, 1, 2])
        tqu = np.asarray(tqu, dtype=np.float64) * float(self.cfg.unit_scale)
        tqu = hp.ud_grade(tqu, self.nside)
        return tqu


class SOCarLoader:
    """Read SO CAR FITS maps, reproject to HEALPix, ud_grade to nside, scale."""

    def __init__(self, cfg: InstrumentConfig, nside: int) -> None:
        self.cfg = cfg
        self.nside = nside

    def load(self, tracer: Tracer) -> np.ndarray:
        import healpy as hp
        from pixell import enmap, reproject

        path = _format_path(self.cfg.map_template, tracer)
        car_map = enmap.read_fits(path)
        hpx = reproject.map2healpix(car_map, method="spline", order=1)
        hpx = np.asarray(hpx, dtype=np.float64) * float(self.cfg.unit_scale)
        hpx = hp.ud_grade(hpx, self.nside)
        return hpx


def get_loader(cfg: InstrumentConfig, nside: int) -> MapLoader:
    """Dispatch on ``cfg.pixelization``."""
    if cfg.pixelization == "healpix_equatorial":
        return PlanckHealpixLoader(cfg, nside)
    if cfg.pixelization == "car":
        return SOCarLoader(cfg, nside)
    raise ValueError(f"unknown pixelization: {cfg.pixelization!r}")
