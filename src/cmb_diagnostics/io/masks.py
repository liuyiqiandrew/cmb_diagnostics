"""Mask dataclass + loaders.

Mask itself is a real frozen dataclass (pure data, no computation). Loaders
port from V1 ``PSContainer.init_mask`` / ``init_mask_from_box`` and the helpers
in ``diag_utils.py``.
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


def load_mask(cfg: MaskConfig, nside: int) -> Mask:
    """Build a HEALPix mask from a :class:`MaskConfig`.

    ``cfg.kind``:
    - ``"file"``: read FITS mask from ``cfg.path``. Dispatch on
      ``cfg.pixelization`` (``"healpix"`` via ``healpy.read_map`` or ``"car"``
      via ``pixell`` + ``reproject.map2healpix``).
    - ``"boxes"``: OR together each ``[[dec_min, ra_min], [dec_max, ra_max]]``
      rectangle from ``cfg.boxes`` (degrees).

    Float vs boolean behavior:
    - ``apodize=True``: raw mask is binarized via ``raw > threshold``, then fed
      through the C2 apodization pipeline. Use for boolean input masks.
    - ``apodize=False``: raw mask is kept as floats, with weights below
      ``threshold`` zeroed in place. Use for pre-apodized analysis masks —
      binarizing would destroy the apodization weights.
    """
    import healpy as hp

    if cfg.kind == "file":
        if cfg.path is None:
            raise ValueError("mask.kind='file' requires mask.path")
        if cfg.pixelization == "healpix":
            raw = hp.read_map(cfg.path)
        elif cfg.pixelization == "car":
            from pixell import enmap, reproject

            car = enmap.read_fits(cfg.path)
            raw = reproject.map2healpix(car, method="spline", order=1)
        else:
            raise ValueError(f"unknown mask.pixelization: {cfg.pixelization!r}")
        raw = np.asarray(hp.ud_grade(raw, nside), dtype=np.float64)
    elif cfg.kind == "boxes":
        if not cfg.boxes:
            raise ValueError("mask.kind='boxes' requires a non-empty mask.boxes list")
        raw = np.zeros(hp.nside2npix(nside), dtype=bool)
        for box in cfg.boxes:
            raw = raw | box2hpmask(nside, np.asarray(box, dtype=float))
    else:
        raise ValueError(f"unknown mask.kind: {cfg.kind!r}")

    if cfg.apodize:
        # Apodization pipeline takes a boolean mask and produces smooth floats.
        bool_raw = raw > cfg.threshold if raw.dtype != bool else raw
        hp_map = apodize_square_mask(bool_raw)
    else:
        # Preserve float weights; zero out sub-threshold pixels in place.
        floats = raw.astype(np.float64, copy=True)
        floats[floats < cfg.threshold] = 0.0
        hp_map = floats
    fsky = effective_fsky(hp_map)
    return Mask(hp_map=np.asarray(hp_map, dtype=np.float64), nside=nside, fsky_effective=fsky)


def box2hpmask(nside: int, box: np.ndarray) -> np.ndarray:
    """Build a boolean HEALPix mask covering one rectangle in (dec, ra) degrees.

    ``box`` must be a 2x2 array: ``[[dec_min, ra_min], [dec_max, ra_max]]``.
    Ported verbatim from ``cmb_diagnoistics/diag_utils.py::box2hpmask``.
    """
    import healpy as hp

    box = np.asarray(box, dtype=float)
    pix_ind = np.arange(hp.nside2npix(nside))
    pix_ang = np.array(hp.pix2ang(nside, pix_ind, lonlat=True)) / 180 * np.pi
    box_rad = box / 180 * np.pi
    if box_rad[1, 1] < 0:
        ra_cut = (pix_ang[0] > (2 * np.pi + box_rad[0, 1])) * (
            pix_ang[0] < (2 * np.pi + box_rad[1, 1])
        )
    else:
        ra_cut = (pix_ang[0] > box_rad[0, 1]) * (pix_ang[0] < box_rad[1, 1])
    dec_cut = (pix_ang[1] > box_rad[0, 0]) * (pix_ang[1] < box_rad[1, 0])
    return ra_cut * dec_cut


def apodize_square_mask(mask: np.ndarray) -> np.ndarray:
    """Smooth + C2-apodize a boolean HEALPix mask.

    Ported verbatim from ``cmb_diagnoistics/diag_utils.py::apodize_square_mask``.
    """
    import healpy as hp
    import pymaster as nmt

    ZERO = 1e-4
    nhg = hp.smoothing(mask.astype(np.float64), 4 / 180 * np.pi)
    nhg[nhg < 0] = 0
    nhg /= nhg.max()
    tmp_mask = nhg > ZERO
    return nmt.mask_apodization(tmp_mask.astype(np.float64), 10, "C2")


def effective_fsky(mask: np.ndarray) -> float:
    """Apodization-weighted effective sky fraction: ``sum(w^2) / Npix``.

    Replaces V1's plain ``sum(w) / Npix``; the squared form is the conventional
    effective fsky for Gaussian (Knox) covariance with an apodized mask.
    """
    m = np.asarray(mask, dtype=np.float64)
    return float((m ** 2).sum() / m.size)
