"""Phase 4 unit tests: primitives, models, dust-fit, estimator round-trips."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from cmb_diagnostics._types import BandInfo, SpectrumKey, Tracer
from cmb_diagnostics.estimators.pol_angle import PolarizationAngleEB
from cmb_diagnostics.estimators.transfer_function import (
    TransferFunctionEE,
    TransferFunctionTE,
)
from cmb_diagnostics.fitting.chi2 import Fitter, fisher_error
from cmb_diagnostics.models.cmb import CMBReference
from cmb_diagnostics.models.dust import MBBDustModel
from cmb_diagnostics.models.tf import (
    RotationModel,
    TFAmplitudeModel,
    rotation_from_amplitude,
)
from cmb_diagnostics.spectra.store import Spectra


# ---- T1. Primitives ----


def test_fitter_recovers_scalar_fit():
    """Minimise chi^2 on y = 2.5*x with noise; Fitter recovers slope."""
    rng = np.random.default_rng(0)
    x = np.linspace(1, 5, 50)
    y = 2.5 * x + rng.normal(0, 0.01, x.size)
    dy = np.full_like(x, 0.01)

    def model(p, xx):
        return p[0] * xx

    fitter = Fitter(model=model, x=x, y=y, dy=dy)
    res = fitter.fit(x0=np.array([1.0]))
    assert abs(float(res.x[0]) - 2.5) < 0.01


def test_fitter_args_tuple_forwarded():
    """The ``args`` tuple is passed to ``model(params, x, args)``."""
    def model(p, x, args):
        scale, = args
        return p[0] * x * scale

    fitter = Fitter(model=model, x=np.array([1.0, 2.0]), y=np.array([2.0, 4.0]),
                    dy=np.array([0.01, 0.01]), args=(2.0,))
    res = fitter.fit(x0=np.array([1.0]))
    assert abs(float(res.x[0]) - 1.0) < 1e-6


def test_fisher_error_matches_closed_form():
    """fisher_error(dm, var) = 1/sqrt(sum(dm**2/var))."""
    dm = np.array([1.0, 2.0, 3.0])
    var = np.array([0.25, 1.0, 4.0])
    expected = 1.0 / np.sqrt((dm ** 2 / var).sum())
    assert fisher_error(dm, var) == pytest.approx(expected)


# ---- T2. TF / rotation models ----


def test_tf_amplitude_predict_scales_linearly():
    tf = TFAmplitudeModel()
    np.testing.assert_allclose(tf.predict(2.0, np.arange(5)), [0, 2, 4, 6, 8])


def test_rotation_predict_scales_linearly():
    rot = RotationModel()
    x = np.array([1.0, 2.0, -3.0])
    np.testing.assert_allclose(rot.predict(0.5, x), 0.5 * x)


def test_rotation_from_amplitude_zero():
    assert rotation_from_amplitude(0.0) == pytest.approx(0.0)


def test_rotation_from_amplitude_matches_closed_form():
    a = 0.1
    assert rotation_from_amplitude(a) == pytest.approx(np.arctan(0.2) / 4)


# ---- Shared synthetic-Spectra helpers ----


def _planck_tracers(freqs=(100.0, 143.0, 217.0, 353.0)) -> list[Tracer]:
    return [Tracer("Planck", f, spin=2) for f in freqs]


def _planck_t_tracers(freqs=(100.0, 143.0, 217.0, 353.0)) -> list[Tracer]:
    return [Tracer("Planck", f, spin=0) for f in freqs]


def _so_tracers(freqs=(90.0, 150.0)) -> list[Tracer]:
    return [Tracer("SO_SAT", f, spin=2) for f in freqs]


_DUST_EFF = {100.0: 105.25, 143.0: 148.235, 217.0: 229.097, 353.0: 372.193,
             90.0: 95.0, 150.0: 155.0}


def _band_info(tracers: list[Tracer]) -> dict[Tracer, BandInfo]:
    return {
        t: BandInfo(tracer=t, beam_fwhm_arcmin=10.0,
                    eff_freq_dust=_DUST_EFF[t.freq])
        for t in tracers
    }


def _bandpowers_small(nbins=5, bin_width=20, lmin=30, lmax=300):
    from cmb_diagnostics.models.bandpowers import Bandpowers

    effective_ell = np.linspace(lmin + bin_width / 2, lmax - bin_width / 2, nbins)
    return Bandpowers(
        nmt_bin=None, bin_width=bin_width, lmin=lmin, lmax=lmax,
        is_dell=True, _effective_ell=effective_ell,
    )


# ---- T3. MBBDustModel.fit_amplitude ----


def test_mbb_fit_amplitude_recovers_injected():
    pytest.importorskip("pygsm")

    bp = _bandpowers_small()
    plk = _planck_tracers()
    dust = MBBDustModel(band_info=_band_info(plk))
    cmb_ref = CMBReference(bandpowers=bp, cls_binned={"EE": np.full(bp.effective_ell.size, 0.5)})

    pairs = list(itertools.combinations(plk, 2))
    a_true = 3.0
    rng = np.random.default_rng(1)

    spec = Spectra(bandpowers=bp)
    for (t1, t2) in pairs:
        dust_unity = dust.predict_cross(t1, t2)
        cl = a_true * dust_unity + cmb_ref.get("EE")
        var = np.full(cl.shape, (cl.mean() * 0.02) ** 2)
        cl = cl + rng.normal(0, np.sqrt(var))
        spec.add(SpectrumKey(t1, t2, "EE"), cl, var)

    fit = dust.fit_amplitude(spec, cmb_ref, pairs, comp="EE", ell_idx=2)
    assert abs(fit.value - a_true) < 3 * fit.error


def test_mbb_fit_amplitude_raises_when_variance_missing():
    pytest.importorskip("pygsm")

    bp = _bandpowers_small()
    plk = _planck_tracers()
    dust = MBBDustModel(band_info=_band_info(plk))
    cmb_ref = CMBReference(bandpowers=bp, cls_binned={"EE": np.full(bp.effective_ell.size, 0.5)})

    pairs = list(itertools.combinations(plk, 2))
    spec = Spectra(bandpowers=bp)
    for (t1, t2) in pairs:
        spec.add(SpectrumKey(t1, t2, "EE"), np.ones(bp.effective_ell.size))  # no var

    with pytest.raises(ValueError, match="no variance"):
        dust.fit_amplitude(spec, cmb_ref, pairs, comp="EE", ell_idx=0)


# ---- T4. TransferFunctionEE synthetic round-trip ----


def _inject_pp_ee(spec: Spectra, dust: MBBDustModel, cmb_ref: CMBReference,
                  tracers: list[Tracer], a_true: float, rng) -> None:
    for (t1, t2) in itertools.product(tracers, tracers):
        dust_unity = dust.predict_cross(t1, t2)
        cl = a_true * dust_unity + cmb_ref.get("EE")
        var = np.full(cl.shape, (cl.mean() * 0.02) ** 2)
        cl = cl + rng.normal(0, np.sqrt(var))
        spec.add(SpectrumKey(t1, t2, "EE"), cl, var)


def _inject_ps_ee(spec: Spectra, dust: MBBDustModel, cmb_ref: CMBReference,
                  plk: list[Tracer], so: list[Tracer],
                  a_true: float, r_true: float, rng) -> None:
    for (pt, st) in itertools.product(plk, so):
        dust_unity = dust.predict_cross(pt, st)
        ref = a_true * dust_unity + cmb_ref.get("EE")
        cl = r_true * ref
        var = np.full(cl.shape, (cl.mean() * 0.02) ** 2)
        cl = cl + rng.normal(0, np.sqrt(var))
        spec.add(SpectrumKey(pt, st, "EE"), cl, var)


def test_transfer_function_ee_recovers_injected_tf():
    pytest.importorskip("pygsm")

    bp = _bandpowers_small()
    plk = _planck_tracers()
    so = _so_tracers()
    band_info = _band_info(plk + so)
    dust = MBBDustModel(band_info=band_info)
    cmb_ref = CMBReference(bandpowers=bp, cls_binned={"EE": np.full(bp.effective_ell.size, 0.5)})

    a_true, r_true = 2.0, 0.8
    rng = np.random.default_rng(42)

    spec_pp = Spectra(bandpowers=bp)
    _inject_pp_ee(spec_pp, dust, cmb_ref, plk, a_true, rng)

    spec_ps = Spectra(bandpowers=bp)
    _inject_ps_ee(spec_ps, dust, cmb_ref, plk, so, a_true, r_true, rng)

    est = TransferFunctionEE(spec_pp=spec_pp, spec_ps=spec_ps, cmb_ref=cmb_ref, dust=dust)
    result = est.estimate(target=Tracer("SO_SAT", 90.0, spin=2))

    assert result.name == "tf_ee_SO_SAT_90"
    assert result.values.shape == result.ell.shape
    # TF = r**2 = 0.64; errors should cover truth for 3-sigma envelope.
    assert np.all(np.abs(result.values - r_true ** 2) < 3 * result.errors + 0.05)
    # r_hat diagnostic mean close to 0.8
    assert abs(result.diagnostics["r"].mean() - r_true) < 0.05


# ---- T5. TransferFunctionTE synthetic ----


def _inject_pp_te(spec: Spectra, dust: MBBDustModel, cmb_ref: CMBReference,
                  plk_t: list[Tracer], plk_e: list[Tracer],
                  a_true: float, rng) -> None:
    for (t0, t2) in itertools.product(plk_t, plk_e):
        dust_unity = dust.predict_cross(t0, t2)
        cl = a_true * dust_unity + cmb_ref.get("TE")
        var = np.full(cl.shape, (abs(cl.mean()) * 0.02 + 1e-9) ** 2)
        cl = cl + rng.normal(0, np.sqrt(var))
        spec.add(SpectrumKey(t0, t2, "TE"), cl, var)


def _inject_ps_te(spec: Spectra, dust: MBBDustModel, cmb_ref: CMBReference,
                  plk_t: list[Tracer], so: list[Tracer],
                  a_true: float, r_true: float, rng) -> None:
    for (pt, st) in itertools.product(plk_t, so):
        dust_unity = dust.predict_cross(pt, st)
        ref = a_true * dust_unity + cmb_ref.get("TE")
        cl = r_true * ref
        var = np.full(cl.shape, (abs(cl.mean()) * 0.02 + 1e-9) ** 2)
        cl = cl + rng.normal(0, np.sqrt(var))
        spec.add(SpectrumKey(pt, st, "TE"), cl, var)


def test_transfer_function_te_recovers_injected_tf():
    pytest.importorskip("pygsm")

    bp = _bandpowers_small(lmin=50)
    plk_t = _planck_t_tracers()
    plk_e = _planck_tracers()
    so = _so_tracers()
    band_info = _band_info(plk_e + so)
    band_info.update(_band_info(plk_t))
    dust = MBBDustModel(band_info=band_info)
    cmb_ref = CMBReference(bandpowers=bp, cls_binned={"TE": np.full(bp.effective_ell.size, 0.5)})

    a_true, r_true = 1.5, 0.9
    rng = np.random.default_rng(7)

    spec_pp = Spectra(bandpowers=bp)
    _inject_pp_te(spec_pp, dust, cmb_ref, plk_t, plk_e, a_true, rng)

    spec_ps = Spectra(bandpowers=bp)
    _inject_ps_te(spec_ps, dust, cmb_ref, plk_t, so, a_true, r_true, rng)

    est = TransferFunctionTE(
        spec_pp_tt=spec_pp, spec_pp_te=spec_pp, spec_ps_te=spec_ps,
        cmb_ref=cmb_ref, dust=dust,
    )
    result = est.estimate(target=Tracer("SO_SAT", 90.0, spin=2))

    assert result.name == "tf_te_SO_SAT_90"
    assert np.all(np.abs(result.values - r_true ** 2) < 3 * result.errors + 0.05)


# ---- T6. PolarizationAngleEB synthetic ----


def test_polarization_angle_eb_recovers_injected_angle():
    bp = _bandpowers_small(nbins=20, bin_width=20, lmin=20, lmax=500)
    so = _so_tracers()

    a_true = 0.02
    alpha_true = rotation_from_amplitude(a_true)
    rng = np.random.default_rng(13)

    spec_ss = Spectra(bandpowers=bp)
    for (t1, t2) in itertools.combinations_with_replacement(so, 2):
        ee = np.full(bp.effective_ell.size, 1.0)
        bb = np.full(bp.effective_ell.size, 0.1)
        eb_signal = a_true * (ee - bb)
        var = np.full(bp.effective_ell.size, 1e-6)
        eb = eb_signal + rng.normal(0, np.sqrt(var))
        spec_ss.add(SpectrumKey(t1, t2, "EE"), ee, np.ones_like(ee) * 1e-4)
        spec_ss.add(SpectrumKey(t1, t2, "BB"), bb, np.ones_like(bb) * 1e-4)
        spec_ss.add(SpectrumKey(t1, t2, "EB"), eb, var)

    est = PolarizationAngleEB(spec_ss=spec_ss, lmin=30, lmax_sweep=(200, 300, 500))
    result = est.estimate()

    assert result.name == "pol_angle_eb"
    assert result.values.shape == (3, 3)  # 3 pairs, 3 caps
    # Recover alpha_true within a few sigma on each (pair, cap).
    assert np.all(np.abs(result.values - alpha_true) < 5 * result.errors + 1e-4)


# ---- T7. Pipeline wiring ----


def test_pipeline_ensure_spectra_raises_without_mask(tiny_config):
    from cmb_diagnostics import Pipeline

    pipe = Pipeline(tiny_config)
    with pytest.raises(RuntimeError, match="load_mask"):
        pipe.estimate_pol_angle()


def test_pipeline_ensure_spectra_raises_without_spectra(phase3_config):
    from cmb_diagnostics import Pipeline

    pipe = Pipeline(phase3_config)
    pipe.load_mask()
    with pytest.raises(RuntimeError, match="compute_spectra"):
        pipe.estimate_tf_ee(target=Tracer("SO_SAT", 90.0, spin=2))


def test_pipeline_build_band_info_covers_all_bands(phase3_config):
    from cmb_diagnostics import Pipeline

    pipe = Pipeline(phase3_config)
    bim = pipe._build_band_info_map()
    # phase3_config has 1 Planck band (143) and 1 SO band (90), both spins.
    assert Tracer("Planck", 143.0, spin=0) in bim
    assert Tracer("Planck", 143.0, spin=2) in bim
    assert Tracer("SO_SAT", 90.0, spin=0) in bim
    assert Tracer("SO_SAT", 90.0, spin=2) in bim
    assert bim[Tracer("Planck", 143.0, spin=2)].eff_freq_dust == pytest.approx(148.2)


def test_pipeline_estimate_tf_ee_with_synthetic_stack(phase3_config):
    """Monkey-patch spectra on the Pipeline and call estimate_tf_ee."""
    pytest.importorskip("pygsm")

    from cmb_diagnostics import Pipeline

    pipe = Pipeline(phase3_config)
    pipe.load_mask()
    # Minimal synthetic Spectra for pp (Planck 143 self-auto) + ps (143 x SO 90).
    # phase3_config has only one Planck band and one SO band, so we need at
    # least two Planck tracers to form a combinations pair. Supplement by
    # adding 100 GHz synthetically here.
    bp = pipe.bandpowers
    assert bp is not None
    cmb_ref = pipe.cmb_ref
    assert cmb_ref is not None

    plk = [Tracer("Planck", 100.0, spin=2), Tracer("Planck", 143.0, spin=2)]
    so = [Tracer("SO_SAT", 90.0, spin=2)]
    band_info = {
        plk[0]: BandInfo(plk[0], 9.66, eff_freq_cmb=101.31, eff_freq_dust=105.25),
        plk[1]: BandInfo(plk[1], 7.27, eff_freq_cmb=142.71, eff_freq_dust=148.235),
        so[0]: BandInfo(so[0], 27.4, eff_freq_dust=95.0),
    }
    dust = MBBDustModel(band_info=band_info)
    a_true, r_true = 1.5, 0.7
    rng = np.random.default_rng(99)

    pp = Spectra(bandpowers=bp)
    _inject_pp_ee(pp, dust, cmb_ref, plk, a_true, rng)
    ps = Spectra(bandpowers=bp)
    _inject_ps_ee(ps, dust, cmb_ref, plk, so, a_true, r_true, rng)
    ss = Spectra(bandpowers=bp)

    pipe.spectra["pp"] = pp
    pipe.spectra["ps"] = ps
    pipe.spectra["ss"] = ss

    # Replace the dust Pipeline builds with one that has matching band_info.
    pipe._build_dust = lambda: dust  # type: ignore[assignment]

    result = pipe.estimate_tf_ee(target=so[0])
    assert result.name == "tf_ee_SO_SAT_90"
    assert "tf_ee_SO_SAT_90" in pipe.results
    assert abs(result.diagnostics["r"].mean() - r_true) < 0.1
