"""Phase 3 unit tests: IO, fields, spectra, models foundations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cmb_diagnostics import Bandpowers, MBBDustModel, Pipeline, Tracer
from cmb_diagnostics._types import BandInfo
from cmb_diagnostics.config import BandpowersConfig, CambConfig, MaskConfig
from cmb_diagnostics.io.camb import load_camb_reference
from cmb_diagnostics.io.masks import (
    Mask,
    apodize_square_mask,
    box2hpmask,
    effective_fsky,
    load_mask,
)
from cmb_diagnostics.spectra.covariance import knox_variance

# ---- masks ----


def test_box2hpmask_basic():
    nside = 16
    box = np.array([[-30.0, -20.0], [30.0, 20.0]])
    m = box2hpmask(nside, box)
    assert m.dtype == bool
    assert m.shape == (12 * nside * nside,)
    assert m.sum() > 0
    assert m.sum() < m.size


def test_effective_fsky_uniform():
    m = np.ones(100)
    assert effective_fsky(m) == pytest.approx(1.0)
    m = np.zeros(100)
    m[:50] = 1.0
    assert effective_fsky(m) == pytest.approx(0.5)


def test_effective_fsky_apodized_differs_from_raw_sum():
    # For a w with 0 < w < 1 somewhere, sum(w^2)/N < sum(w)/N.
    m = np.array([1.0, 0.5, 0.25, 0.0])
    assert effective_fsky(m) < m.sum() / m.size


def test_load_mask_boxes_no_apodize():
    cfg = MaskConfig(
        kind="boxes",
        apodize=False,
        threshold=1e-5,
        boxes=[[[-30.0, -20.0], [30.0, 20.0]]],
    )
    mask = load_mask(cfg, nside=16)
    assert isinstance(mask, Mask)
    assert mask.nside == 16
    assert mask.hp_map.shape == (12 * 16 * 16,)
    assert 0 < mask.fsky_effective < 1


def test_load_mask_rejects_unknown_kind():
    cfg = MaskConfig(kind="nonsense", apodize=False, boxes=None)
    with pytest.raises(ValueError):
        load_mask(cfg, nside=16)


def test_apodize_square_mask_returns_smooth_floats():
    import healpy as hp

    nside = 16
    raw = np.zeros(hp.nside2npix(nside), dtype=bool)
    box = np.array([[-30.0, -20.0], [30.0, 20.0]])
    raw |= box2hpmask(nside, box)
    ap = apodize_square_mask(raw)
    assert ap.dtype == np.float64
    # Apodized mask is strictly softer than the raw boolean mask.
    assert ap.max() <= 1.0 + 1e-12
    assert ap.min() >= 0.0 - 1e-12
    assert np.any((ap > 0) & (ap < 1))


# ---- bandpowers ----


def test_bandpowers_from_config_populates_nmt_bin():
    cfg = BandpowersConfig(bin_width=20, lmin=30, lmax=100, is_Dell=True)
    bp = Bandpowers.from_config(cfg, nside=16)
    assert bp.nmt_bin is not None
    assert bp.bin_width == 20
    assert bp.is_dell is True
    eff = bp.effective_ell
    assert eff.ndim == 1
    assert eff.size > 0
    # dl2cl = 2*pi/l/(l+1)
    expected = 2 * np.pi / eff / (eff + 1)
    np.testing.assert_allclose(bp.dl2cl, expected)
    # msk selects bins strictly inside (lmin, lmax)
    assert bp.msk.dtype == bool
    assert bp.msk.any()


# ---- knox ----


def test_knox_variance_symmetric_matches_formula(bandpowers_factory):
    bp = bandpowers_factory(nbins=5, bin_width=20, lmin=30, lmax=300)
    n = bp.effective_ell.size
    cl_aa = np.full(n, 2.0)
    cl_bb = np.full(n, 3.0)
    cl_ab = np.full(n, 1.0)
    fsky = 0.2
    var = knox_variance(cl_aa, cl_bb, cl_ab, bp, fsky)
    expected = (cl_aa * cl_bb + cl_ab * cl_ab) / (2 * bp.effective_ell + 1) / fsky / bp.bin_width
    np.testing.assert_allclose(var, expected)


def test_knox_variance_cross_term_uses_cl_ba(bandpowers_factory):
    bp = bandpowers_factory(nbins=5)
    n = bp.effective_ell.size
    cl_aa = np.full(n, 2.0)
    cl_bb = np.full(n, 3.0)
    cl_ab = np.full(n, 0.5)
    cl_ba = np.full(n, 0.7)
    fsky = 0.3
    var = knox_variance(cl_aa, cl_bb, cl_ab, bp, fsky, cl_ba=cl_ba)
    expected = (cl_aa * cl_bb + cl_ab * cl_ba) / (2 * bp.effective_ell + 1) / fsky / bp.bin_width
    np.testing.assert_allclose(var, expected)


# ---- CAMB ----


def test_load_camb_reference_roundtrip(synthetic_camb_dat: Path):
    cfg = CambConfig(path=str(synthetic_camb_dat), format="bbpower_nobb")
    bp = Bandpowers.from_config(BandpowersConfig(bin_width=10, lmin=5, lmax=40, is_Dell=True), nside=16)
    ref = load_camb_reference(cfg, bp, nside=16)
    for comp in ("TT", "EE", "BB", "TE"):
        assert ref.has(comp)
        assert ref.get(comp).shape == bp.effective_ell.shape
    # synthetic BB column is zero -> binned BB is zero.
    np.testing.assert_allclose(ref.get("BB"), 0.0)


def test_load_camb_reference_unknown_format_raises(synthetic_camb_dat: Path):
    cfg = CambConfig(path=str(synthetic_camb_dat), format="weird")
    bp = Bandpowers.from_config(BandpowersConfig(bin_width=10, lmin=5, lmax=40, is_Dell=True), nside=16)
    with pytest.raises(ValueError):
        load_camb_reference(cfg, bp, nside=16)


# ---- MBB dust ----


def test_mbb_predict_cross_matches_v1_formula():
    pygsm = pytest.importorskip("pygsm")
    t1 = Tracer("Planck", 100.0, spin=2)
    t2 = Tracer("Planck", 143.0, spin=2)
    band_info = {
        t1: BandInfo(t1, beam_fwhm_arcmin=9.66, eff_freq_dust=105.25),
        t2: BandInfo(t2, beam_fwhm_arcmin=7.27, eff_freq_dust=148.235),
    }
    dust = MBBDustModel(beta=1.6, Td_kelvin=19.6, nu0_ghz=353.0, band_info=band_info)
    got = dust.predict_cross(t1, t2)

    f1, f2, nu0 = 105.25, 148.235, 353.0
    beta, Td = 1.6, 19.6
    expected = (
        (f1 * f2 / nu0 ** 2) ** beta
        * pygsm.planck_law(Td, f1) * pygsm.planck_law(Td, f2) / pygsm.planck_law(Td, nu0) ** 2
        * pygsm.trj2tcmb(f1) * pygsm.trj2tcmb(f2)
    )
    assert got == pytest.approx(expected)


def test_mbb_predict_cross_missing_band_raises():
    t1 = Tracer("Planck", 100.0, spin=2)
    t2 = Tracer("Planck", 143.0, spin=2)
    dust = MBBDustModel()
    with pytest.raises(KeyError):
        dust.predict_cross(t1, t2)


# ---- pipeline Phase 3 steps ----


def test_pipeline_phase3_steps_succeed(phase3_config):
    pipe = Pipeline(phase3_config)
    mask = pipe.load_mask()
    assert isinstance(mask, Mask)
    assert pipe.bandpowers is not None
    assert pipe.cmb_ref is not None
    assert pipe.cmb_ref.has("EE")


def test_pipeline_build_fieldsets_requires_mask(phase3_config):
    pipe = Pipeline(phase3_config)
    with pytest.raises(RuntimeError, match="load_mask"):
        pipe.build_fieldsets()


def test_pipeline_compute_spectra_requires_fieldsets(phase3_config):
    pipe = Pipeline(phase3_config)
    pipe.load_mask()
    with pytest.raises(RuntimeError, match="build_fieldsets"):
        pipe.compute_spectra()
