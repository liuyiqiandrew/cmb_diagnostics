"""Build NaMaster fields from raw maps + beams + mask."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from cmb_diagnostics._types import Tracer
from cmb_diagnostics.fields.container import FieldSet
from cmb_diagnostics.io.loaders import get_loader

if TYPE_CHECKING:
    from cmb_diagnostics.config import InstrumentConfig
    from cmb_diagnostics.io.masks import Mask


def _gauss_beam(fwhm_arcmin: float, nside: int) -> np.ndarray:
    import healpy as hp

    # V1 used `nside*3 - 1` for Planck, `nside*3` for SO in the test. Use the
    # +0 form consistently (one extra ell costs nothing, and V2's
    # `new_estimator_test.py` uses `nside*3` for both).
    return hp.gauss_beam(fwhm_arcmin / 60 / 180 * np.pi, nside * 3)


def build_spin0_field(
    tmap: np.ndarray,
    mask: Mask,
    beam_fwhm_arcmin: float,
    nside: int,
) -> Any:
    """Spin-0 NaMaster field from a single-component temperature map."""
    import pymaster as nmt

    beam = _gauss_beam(beam_fwhm_arcmin, nside)
    return nmt.NmtField(mask.hp_map, [np.asarray(tmap)], beam=beam, spin=0)


def build_spin2_field(
    qmap: np.ndarray,
    umap: np.ndarray,
    mask: Mask,
    beam_fwhm_arcmin: float,
    nside: int,
    purify_e: bool = False,
    purify_b: bool = False,
) -> Any:
    """Spin-2 NaMaster field from Q/U maps."""
    import pymaster as nmt

    beam = _gauss_beam(beam_fwhm_arcmin, nside)
    return nmt.NmtField(
        mask.hp_map,
        [np.asarray(qmap), np.asarray(umap)],
        beam=beam,
        purify_e=purify_e,
        purify_b=purify_b,
    )


def build_fieldset(cfg: InstrumentConfig, mask: Mask, nside: int) -> FieldSet:
    """Load I/Q/U per band, construct spin-0 and spin-2 NaMaster fields."""
    loader = get_loader(cfg, nside)
    fs = FieldSet(cfg.name)
    for band in cfg.bands:
        tqu = loader.load(Tracer(cfg.name, band.freq, spin=2))
        if tqu.ndim != 2 or tqu.shape[0] < 3:
            raise RuntimeError(
                f"{cfg.name} loader returned shape {tqu.shape}; expected (3, npix)."
            )
        t0 = build_spin0_field(tqu[0], mask, band.beam_fwhm_arcmin, nside)
        t2 = build_spin2_field(
            tqu[1],
            tqu[2],
            mask,
            band.beam_fwhm_arcmin,
            nside,
            purify_e=cfg.purify_e,
            purify_b=cfg.purify_b,
        )
        fs.add(Tracer(cfg.name, band.freq, spin=0), t0)
        fs.add(Tracer(cfg.name, band.freq, spin=2), t2)
    return fs
