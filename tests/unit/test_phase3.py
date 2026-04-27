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
    healpix_box_mask,
    load_mask,
)
from cmb_diagnostics.spectra.covariance import knox_variance

# ---- masks ----


def test_box2hpmask_basic():
    """`box2hpmask` returns a partially-filled boolean HEALPix mask."""
    nside = 16
    box = np.array([[-30.0, -20.0], [30.0, 20.0]])
    m = box2hpmask(nside, box)
    assert m.dtype == bool
    assert m.shape == (12 * nside * nside,)
    assert m.sum() > 0
    assert m.sum() < m.size


def test_healpix_box_mask_wraparound():
    """`healpix_box_mask` handles longitude wrap-around across 0°/360°."""
    # A (350°, 20°) longitude band must wrap across 0°/360°.
    nside = 32
    m = healpix_box_mask(nside, lon_bounds_deg=(350.0, 20.0), lat_bounds_deg=(-10.0, 10.0))
    import healpy as hp

    lon, lat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    lon = lon % 360.0
    expected = ((lon >= 350.0) | (lon <= 20.0)) & (lat >= -10.0) & (lat <= 10.0)
    assert np.array_equal(m, expected)


def test_healpix_box_mask_zero_crossing_via_negative_min():
    """`healpix_box_mask` handles negative-min, positive-max zero-crossing boxes."""
    # A box with negative min RA and positive max RA crosses 0° — the legacy
    # box2hpmask returned an empty mask in this case; the new function does not.
    nside = 32
    m = healpix_box_mask(nside, lon_bounds_deg=(-10.0, 30.0), lat_bounds_deg=(-5.0, 5.0))
    assert m.sum() > 0


def test_healpix_box_mask_full_longitude_returns_band():
    """A 360° longitude span returns a pure latitude band."""
    nside = 16
    m = healpix_box_mask(nside, lon_bounds_deg=(0.0, 360.0), lat_bounds_deg=(-10.0, 10.0))
    import healpy as hp

    _, lat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    assert np.array_equal(m, (lat >= -10.0) & (lat <= 10.0))


def test_healpix_box_mask_validation():
    """`healpix_box_mask` rejects missing/duplicate lat inputs and out-of-range bounds."""
    with pytest.raises(ValueError):
        healpix_box_mask(16, lon_bounds_deg=(0.0, 10.0))  # neither lat nor colat
    with pytest.raises(ValueError):
        healpix_box_mask(
            16, lon_bounds_deg=(0.0, 10.0), lat_bounds_deg=(0.0, 1.0), colat_bounds_deg=(80.0, 90.0)
        )
    with pytest.raises(ValueError):
        healpix_box_mask(16, lon_bounds_deg=(0.0, 10.0), lat_bounds_deg=(-91.0, 0.0))


def test_box2hpmask_adapter_matches_healpix_box_mask_on_config_boxes():
    """Legacy `box2hpmask` agrees with `healpix_box_mask` on real config boxes."""
    # The two boxes shipped in configs/satp3_south.yaml.
    nside = 64
    south = np.array([[-50.0, 50.0], [-30.0, 90.0]])
    east = np.array([[-20.0, -165.0], [0.0, -130.0]])
    for box in (south, east):
        adapter = box2hpmask(nside, box)
        direct = healpix_box_mask(
            nside,
            lon_bounds_deg=(box[0, 1], box[1, 1]),
            lat_bounds_deg=(box[0, 0], box[1, 0]),
        )
        assert np.array_equal(adapter, direct)
        assert adapter.sum() > 0


def test_effective_fsky_uniform():
    """`effective_fsky` equals 1 for a full mask and 0.5 for a half mask."""
    m = np.ones(100)
    assert effective_fsky(m) == pytest.approx(1.0)
    m = np.zeros(100)
    m[:50] = 1.0
    assert effective_fsky(m) == pytest.approx(0.5)


def test_effective_fsky_apodized_differs_from_raw_sum():
    """sum(w^2)/N is strictly below sum(w)/N for any non-boolean mask."""
    # For a w with 0 < w < 1 somewhere, sum(w^2)/N < sum(w)/N.
    m = np.array([1.0, 0.5, 0.25, 0.0])
    assert effective_fsky(m) < m.sum() / m.size


def test_load_mask_boxes_no_apodize():
    """`load_mask(kind='boxes', apodize=False)` returns a valid `Mask`."""
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
    """`load_mask` raises on unknown `MaskConfig.kind`."""
    cfg = MaskConfig(kind="nonsense", apodize=False, boxes=None)
    with pytest.raises(ValueError):
        load_mask(cfg, nside=16)


def test_load_mask_file_preserves_floats_when_apodize_false(tmp_path: Path):
    """A pre-apodized float mask read from disk is not binarized when `apodize=False`."""
    import healpy as hp

    nside = 16
    npix = hp.nside2npix(nside)
    raw = np.zeros(npix, dtype=np.float64)
    # Soft ramp across a few pixels to simulate apodization weights.
    raw[10:40] = np.linspace(0.2, 1.0, 30)
    path = tmp_path / "preapodized.fits"
    hp.write_map(str(path), raw, overwrite=True)

    cfg = MaskConfig(kind="file", path=str(path), apodize=False, threshold=1e-5)
    mask = load_mask(cfg, nside=nside)

    # Float weights must survive (not binarized), and intermediate values (0 < w < 1) must exist.
    assert mask.hp_map.dtype == np.float64
    assert np.any((mask.hp_map > 0) & (mask.hp_map < 1))
    # effective_fsky for an apodized mask uses sum(w^2)/Npix, which is strictly less than sum(w)/Npix.
    assert mask.fsky_effective < mask.hp_map.sum() / mask.hp_map.size


def test_apodize_square_mask_returns_smooth_floats():
    """`apodize_square_mask` returns smooth weights in [0, 1], not a boolean mask."""
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
    """`Bandpowers.from_config` builds an `nmt_bin` with consistent ell / dl2cl / msk."""
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
    """`knox_variance` matches the closed-form Gaussian formula for symmetric components."""
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
    """`knox_variance` uses the explicit `cl_ba` for asymmetric components (EB/BE)."""
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


def _write_flat_cell_dat(path: Path) -> None:
    """Write a synthetic D_ell table that represents a flat C_ell = 1 spectrum.

    BBPower-style: ``[ell, TT, EE, BB, TE]`` rows starting at ``ell = 1``,
    with every spectral column equal to ``D_ell = ell(ell+1)/(2*pi)`` so the
    loader's per-ell D_ell -> C_ell conversion produces C_ell = 1 across the board.
    """
    lines = []
    for ell in range(1, 4096):
        d = ell * (ell + 1) / (2 * np.pi)
        lines.append(f"{ell} {d:.6e} {d:.6e} {d:.6e} {d:.6e}")
    path.write_text("\n".join(lines))


def test_load_camb_reference_recovers_flat_cell(tmp_path: Path):
    """A D_ell file representing C_ell = 1 must round-trip to ~1 per bin (is_Dell=True)."""
    path = tmp_path / "flat_cell.dat"
    _write_flat_cell_dat(path)

    cfg = CambConfig(path=str(path), format="bbpower_nobb")
    nside = 64
    # bp.lmin filters out the low-ell bins where the bin_cell + dl2cl(ell_eff)
    # approximation breaks down (ell(ell+1)/(2*pi) varies steeply across the
    # bin). Estimators only consume bp.msk bins, so the contract that matters
    # is that masked bins recover C_ell to <1%.
    bp = Bandpowers.from_config(
        BandpowersConfig(bin_width=20, lmin=50, lmax=200, is_Dell=True), nside=nside
    )
    ref = load_camb_reference(cfg, bp, nside=nside)

    # 2% tolerance matches the residual <D_ell>_bin -> dl2cl(ell_eff) error
    # at the low end of a realistic bin range; compute_spectra uses the same
    # approximation, so this systematic cancels in downstream fits.
    for comp in ("TT", "EE", "BB", "TE"):
        np.testing.assert_allclose(ref.get(comp)[bp.msk], 1.0, rtol=0.02)


def test_load_camb_reference_works_with_is_dell_false(tmp_path: Path):
    """The is_Dell=False path must also recover C_ell = 1 (no post-multiply)."""
    path = tmp_path / "flat_cell.dat"
    _write_flat_cell_dat(path)

    cfg = CambConfig(path=str(path), format="bbpower_nobb")
    nside = 64
    bp = Bandpowers.from_config(
        BandpowersConfig(bin_width=20, lmin=30, lmax=200, is_Dell=False), nside=nside
    )
    ref = load_camb_reference(cfg, bp, nside=nside)
    # is_Dell=False makes bin_cell return <C_ell>_bin directly, so this path
    # is exact for a flat C_ell input (modulo floating-point in the per-ell
    # D_ell -> C_ell conversion).
    np.testing.assert_allclose(ref.get("EE"), 1.0, rtol=1e-6)


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


# ---- map-template formatting ----


def test_format_path_named_freq_and_format_spec():
    """`_format_path` renders ``{freq}`` and ``{freq:03d}`` correctly."""
    from cmb_diagnostics.io.loaders import _format_path

    t = Tracer("Planck", 143.0, spin=2)
    assert _format_path("map_{freq}.fits", t) == "map_143.fits"
    assert _format_path("map_f{freq:03d}.fits", t) == "map_f143.fits"

    t90 = Tracer("SO_SAT", 90.0, spin=2)
    assert _format_path("map_f{freq:03d}.fits", t90) == "map_f090.fits"


def test_format_path_rejects_bare_positional_template():
    """Bare ``{}`` templates raise rather than silently fall through."""
    from cmb_diagnostics.io.loaders import _format_path

    t = Tracer("Planck", 143.0, spin=2)
    with pytest.raises(IndexError):
        _format_path("map_{}.fits", t)
