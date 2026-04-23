"""Mask dataclass + loaders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

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


def _parse_bounds(bounds: Sequence[float], name: str) -> tuple[float, float]:
    arr = np.asarray(bounds, dtype=float).reshape(-1)
    if arr.size != 2 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a length-2 sequence of finite floats.")
    return float(arr[0]), float(arr[1])


def healpix_box_mask(
    nside: int,
    lon_bounds_deg: Sequence[float],
    lat_bounds_deg: Optional[Sequence[float]] = None,
    colat_bounds_deg: Optional[Sequence[float]] = None,
    *,
    nest: bool = False,
    inclusive: bool = True,
) -> np.ndarray:
    """Return a boolean HEALPix mask for a longitude/latitude box.

    Wrap-around in longitude (e.g. ``(350, 20)``) and zero-crossing boxes
    (e.g. negative-min/positive-max) are handled correctly. Provide exactly
    one of ``lat_bounds_deg`` or ``colat_bounds_deg``. Membership is decided
    from pixel-center coordinates from :func:`healpy.pix2ang`.
    """
    import healpy as hp

    if (lat_bounds_deg is None) == (colat_bounds_deg is None):
        raise ValueError("Provide exactly one of lat_bounds_deg or colat_bounds_deg.")
    if not hp.isnsideok(nside):
        raise ValueError(f"Invalid nside={nside}.")

    lon_start, lon_stop = _parse_bounds(lon_bounds_deg, "lon_bounds_deg")
    lon_width = abs(lon_stop - lon_start)
    lon_min = lon_start % 360.0
    lon_max = lon_stop % 360.0
    full_longitude = lon_width > 360.0 or np.isclose(lon_width, 360.0)

    if lat_bounds_deg is not None:
        lat_start, lat_stop = _parse_bounds(lat_bounds_deg, "lat_bounds_deg")
    else:
        colat_start, colat_stop = _parse_bounds(colat_bounds_deg, "colat_bounds_deg")
        if not (0.0 <= colat_start <= 180.0 and 0.0 <= colat_stop <= 180.0):
            raise ValueError("colat_bounds_deg values must lie in [0, 180].")
        lat_start = 90.0 - colat_start
        lat_stop = 90.0 - colat_stop

    lat_min = min(lat_start, lat_stop)
    lat_max = max(lat_start, lat_stop)
    if lat_min < -90.0 or lat_max > 90.0:
        raise ValueError("Latitude bounds must lie in [-90, 90].")

    npix = hp.nside2npix(nside)
    lon_deg, lat_deg = hp.pix2ang(nside, np.arange(npix), nest=nest, lonlat=True)
    lon_deg = lon_deg % 360.0

    if full_longitude:
        lon_mask = np.ones(npix, dtype=bool)
    elif lon_min <= lon_max:
        lon_mask = (lon_deg >= lon_min) & (lon_deg <= lon_max)
    else:
        lon_mask = (lon_deg >= lon_min) | (lon_deg <= lon_max)

    lat_mask = (lat_deg >= lat_min) & (lat_deg <= lat_max)
    mask = lon_mask & lat_mask
    return mask if inclusive else ~mask


def box2hpmask(nside: int, box: np.ndarray) -> np.ndarray:
    """Build a boolean HEALPix mask covering one ``[[dec_min, ra_min], [dec_max, ra_max]]`` rectangle in degrees.

    Adapter over :func:`healpix_box_mask` that preserves the legacy 2x2 box
    schema used by :class:`cmb_diagnostics.config.MaskConfig` and YAML configs.
    """
    box = np.asarray(box, dtype=float)
    if box.shape != (2, 2):
        raise ValueError(f"box must be shape (2, 2), got {box.shape}")
    return healpix_box_mask(
        nside,
        lon_bounds_deg=(box[0, 1], box[1, 1]),
        lat_bounds_deg=(box[0, 0], box[1, 0]),
    )


def apodize_square_mask(mask: np.ndarray) -> np.ndarray:
    """Smooth + C2-apodize a boolean HEALPix mask."""
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
