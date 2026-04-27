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


def test_tf_ee_nan_dust_propagates_nan_not_one():
    """A NaN-returning dust model must yield NaN TF bins, not the optimizer's r=1 guess."""
    pytest.importorskip("pygsm")

    from cmb_diagnostics.models.dust import FitAmplitude

    class _NanDust:
        def predict_cross(self, t1, t2, band_info=None):
            return 1.0

        def fit_amplitude(self, spectra, cmb_ref, tracer_pairs, comp,
                          ell_idx, positive_only=None):
            return FitAmplitude(
                value=float("nan"), error=float("nan"), chi2=float("nan")
            )

    bp = _bandpowers_small()
    plk = _planck_tracers()
    so = _so_tracers()
    real_dust = MBBDustModel(band_info=_band_info(plk + so))
    cmb_ref = CMBReference(
        bandpowers=bp, cls_binned={"EE": np.full(bp.effective_ell.size, 0.5)}
    )

    rng = np.random.default_rng(0)
    spec_pp = Spectra(bandpowers=bp)
    _inject_pp_ee(spec_pp, real_dust, cmb_ref, plk, a_true=2.0, rng=rng)
    spec_ps = Spectra(bandpowers=bp)
    _inject_ps_ee(spec_ps, real_dust, cmb_ref, plk, so,
                  a_true=2.0, r_true=0.8, rng=rng)

    est = TransferFunctionEE(
        spec_pp=spec_pp, spec_ps=spec_ps, cmb_ref=cmb_ref, dust=_NanDust(),
    )
    result = est.estimate(target=Tracer("SO_SAT", 90.0, spin=2))

    assert np.all(np.isnan(result.values)), result.values
    assert np.all(np.isnan(result.errors))


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


def test_tf_te_respects_bandpowers_lmin():
    """TransferFunctionTE must not dip below bp.lmin even when its own lmin is looser."""
    pytest.importorskip("pygsm")

    # bp.lmin=100 is stricter than TE default lmin=50.
    bp = _bandpowers_small(nbins=15, bin_width=20, lmin=100, lmax=400)
    plk_t = _planck_t_tracers()
    plk_e = _planck_tracers()
    so = _so_tracers()
    band_info = _band_info(plk_e + so)
    band_info.update(_band_info(plk_t))
    dust = MBBDustModel(band_info=band_info)
    cmb_ref = CMBReference(
        bandpowers=bp, cls_binned={"TE": np.full(bp.effective_ell.size, 0.5)}
    )

    rng = np.random.default_rng(1)
    spec_pp = Spectra(bandpowers=bp)
    _inject_pp_te(spec_pp, dust, cmb_ref, plk_t, plk_e, a_true=1.5, rng=rng)
    spec_ps = Spectra(bandpowers=bp)
    _inject_ps_te(spec_ps, dust, cmb_ref, plk_t, so,
                  a_true=1.5, r_true=0.9, rng=rng)

    est = TransferFunctionTE(
        spec_pp_tt=spec_pp, spec_pp_te=spec_pp, spec_ps_te=spec_ps,
        cmb_ref=cmb_ref, dust=dust, lmin=50,  # intentionally looser than bp.lmin
    )
    result = est.estimate(target=Tracer("SO_SAT", 90.0, spin=2))

    assert result.ell.size > 0
    assert result.ell.min() > 100, (
        f"TE used bin at ell={result.ell.min()} below bp.lmin=100"
    )


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

    # 2-D HTML repr must not raise (bug 4 regression).
    html = result._repr_html_()
    assert "<table" in html and "</table>" in html


def test_pol_angle_empty_window_returns_nan():
    """lmax below smallest effective_ell must NaN-fill, not leak the x0=1 guess."""
    bp = _bandpowers_small(nbins=10, bin_width=20, lmin=100, lmax=300)
    so = _so_tracers()

    spec_ss = Spectra(bandpowers=bp)
    for (t1, t2) in itertools.combinations_with_replacement(so, 2):
        n = bp.effective_ell.size
        ee = np.ones(n)
        bb = np.full(n, 0.1)
        eb = np.zeros(n)
        spec_ss.add(SpectrumKey(t1, t2, "EE"), ee, np.full(n, 1e-4))
        spec_ss.add(SpectrumKey(t1, t2, "BB"), bb, np.full(n, 1e-4))
        spec_ss.add(SpectrumKey(t1, t2, "EB"), eb, np.full(n, 1e-6))

    # lmax=50 is below every effective_ell (they start above 100), so cap_msk is empty.
    est = PolarizationAngleEB(spec_ss=spec_ss, lmin=30, lmax_sweep=(50,))
    result = est.estimate()

    assert np.all(np.isnan(result.values)), result.values
    assert np.all(np.isnan(result.errors))


# ---- T6.5. compute_spectra D_l -> C_l conversion ----


def test_compute_spectra_converts_dell_to_cell(monkeypatch):
    """compute_spectra applies bandpowers.dl2cl when is_dell=True and not when False.

    With is_dell=True, NaMaster's compute_full_master returns D_l = l(l+1)/(2pi) * C_l;
    compute_spectra must divide by that factor so downstream Spectra are in C_l. This
    test stubs compute_full_master and asserts the conversion is applied.
    """
    pytest.importorskip("pymaster")
    from cmb_diagnostics.fields.container import FieldSet
    from cmb_diagnostics.models.bandpowers import Bandpowers
    from cmb_diagnostics.spectra.compute import compute_spectra

    nbins = 5
    eff_ell = np.array([50.0, 100.0, 200.0, 300.0, 400.0])
    dl2cl = 2 * np.pi / eff_ell / (eff_ell + 1)

    common = dict(
        nmt_bin=object(), bin_width=20, lmin=10, lmax=500,
        _effective_ell=eff_ell,
    )
    bp_dell = Bandpowers(is_dell=True, **common)
    bp_cl = Bandpowers(is_dell=False, **common)

    raw = 1.5
    canned4 = np.full((4, nbins), raw)  # spin2 x spin2
    import pymaster as nmt

    def fake_cfm(f1, f2, nmt_bin):
        return canned4

    monkeypatch.setattr(nmt, "compute_full_master", fake_cfm)

    plk_e = Tracer("Planck", 100.0, spin=2)
    fs = FieldSet("Planck")
    fs.add(plk_e, object())  # any sentinel; cfm is stubbed

    spec_dell = compute_spectra(fs, fs, bp_dell, fsky_effective=0.3)
    spec_cl = compute_spectra(fs, fs, bp_cl, fsky_effective=0.3)

    cl_dell, var_dell = spec_dell.get(plk_e, plk_e, "EE")
    cl_cl, var_cl = spec_cl.get(plk_e, plk_e, "EE")

    np.testing.assert_allclose(cl_dell, raw * dl2cl)
    np.testing.assert_allclose(cl_cl, np.full(nbins, raw))

    # Knox variance scales as C_l^2 in the C_l path, D_l^2 in the unconverted path.
    np.testing.assert_allclose(var_dell, var_cl * dl2cl ** 2)


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
