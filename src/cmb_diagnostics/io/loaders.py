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
    """Structural protocol for the per-instrument map loader.

    Implementations are dispatched by :func:`get_loader` based on
    ``InstrumentConfig.pixelization``. Each must yield I/Q/U HEALPix maps in
    muK at the pipeline's target ``nside`` so downstream code never sees CAR
    geometry or raw K units.
    """

    def load(self, tracer: Tracer) -> np.ndarray:
        """Return a ``(3, npix)`` I/Q/U HEALPix map at the configured nside."""
        ...


def _format_path(template: str, tracer: Tracer) -> str:
    """Format a map path template with the tracer's integer frequency.

    Templates must use the named placeholder ``{freq}`` (e.g.
    ``map_{freq}.fits`` or ``map_f{freq:03d}.fits``).
    """
    return template.format(freq=int(tracer.freq))


class PlanckHealpixLoader:
    """Load Planck equatorial HEALPix T/Q/U FITS maps.

    Reads T, Q, U fields from a single FITS, ``ud_grade``-s to ``nside``, and
    applies ``cfg.unit_scale`` (defaults to ``1e6``; K -> muK).

    Parameters
    ----------
    cfg : InstrumentConfig
        Instrument configuration; ``cfg.map_template`` supplies the path
        template.
    nside : int
        Target HEALPix resolution.
    """

    def __init__(self, cfg: InstrumentConfig, nside: int) -> None:
        self.cfg = cfg
        self.nside = nside

    def load(self, tracer: Tracer) -> np.ndarray:
        """Load a T/Q/U HEALPix triplet for ``tracer``.

        Parameters
        ----------
        tracer : Tracer
            Identifier; ``tracer.freq`` drives the path template.

        Returns
        -------
        numpy.ndarray
            ``(3, npix)`` array in muK at the configured ``nside``.
        """
        import healpy as hp

        path = _format_path(self.cfg.map_template, tracer)
        tqu = hp.read_map(path, field=[0, 1, 2])
        tqu = np.asarray(tqu, dtype=np.float64) * float(self.cfg.unit_scale)
        tqu = hp.ud_grade(tqu, self.nside)
        return tqu


class SOCarLoader:
    """Load SO CAR FITS maps and reproject to HEALPix.

    Reads a CAR map via ``pixell.enmap``, reprojects to HEALPix using spline
    order 1, ``ud_grade``-s to ``nside``, and applies ``cfg.unit_scale``.

    Parameters
    ----------
    cfg : InstrumentConfig
        Instrument configuration; ``cfg.map_template`` supplies the path
        template.
    nside : int
        Target HEALPix resolution.
    """

    def __init__(self, cfg: InstrumentConfig, nside: int) -> None:
        self.cfg = cfg
        self.nside = nside

    def load(self, tracer: Tracer) -> np.ndarray:
        """Load a T/Q/U HEALPix triplet for ``tracer`` from a CAR FITS.

        Parameters
        ----------
        tracer : Tracer
            Identifier; ``tracer.freq`` drives the path template.

        Returns
        -------
        numpy.ndarray
            ``(3, npix)`` array in muK at the configured ``nside``.
        """
        import healpy as hp
        from pixell import enmap, reproject

        path = _format_path(self.cfg.map_template, tracer)
        car_map = enmap.read_fits(path)
        hpx = reproject.map2healpix(car_map, method="spline", order=1)
        hpx = np.asarray(hpx, dtype=np.float64) * float(self.cfg.unit_scale)
        hpx = hp.ud_grade(hpx, self.nside)
        return hpx


def get_loader(cfg: InstrumentConfig, nside: int) -> MapLoader:
    """Select a :class:`MapLoader` based on ``cfg.pixelization``.

    Parameters
    ----------
    cfg : InstrumentConfig
        Instrument configuration.
    nside : int
        Target HEALPix resolution.

    Returns
    -------
    MapLoader
        :class:`PlanckHealpixLoader` for ``"healpix_equatorial"`` or
        :class:`SOCarLoader` for ``"car"``.

    Raises
    ------
    ValueError
        On unknown ``cfg.pixelization``.
    """
    if cfg.pixelization == "healpix_equatorial":
        return PlanckHealpixLoader(cfg, nside)
    if cfg.pixelization == "car":
        return SOCarLoader(cfg, nside)
    raise ValueError(f"unknown pixelization: {cfg.pixelization!r}")
