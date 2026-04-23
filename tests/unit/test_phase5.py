"""Phase 5 unit tests: Phase 4 bug fixes + Pipeline.run + reports + CLI.

Tests are grouped by the plan section (B1-B5 cleanup, C1/C3/C4/C5 Phase 5).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from cmb_diagnostics import (
    CMBReference,
    MBBDustModel,
    Pipeline,
    Spectra,
    TransferFunctionTE,
)
from cmb_diagnostics._types import BandInfo, SpectrumKey, Tracer
from cmb_diagnostics.estimators.pol_angle import PolarizationAngleEB
from cmb_diagnostics.fitting.chi2 import fisher_error
from cmb_diagnostics.models.bandpowers import Bandpowers


# ---- Helpers (borrowed shape from test_phase4.py) ----


_DUST_EFF = {
    100.0: 105.25, 143.0: 148.235, 217.0: 229.097, 353.0: 372.193,
    90.0: 95.0, 150.0: 155.0,
}


def _bandpowers(nbins: int, bin_width: int, lmin: int, lmax: int) -> Bandpowers:
    effective_ell = np.linspace(lmin + bin_width / 2, lmax - bin_width / 2, nbins)
    return Bandpowers(
        nmt_bin=None, bin_width=bin_width, lmin=lmin, lmax=lmax,
        is_dell=True, _effective_ell=effective_ell,
    )


def _planck_e(freqs=(100.0, 143.0, 217.0, 353.0)) -> list[Tracer]:
    return [Tracer("Planck", f, spin=2) for f in freqs]


def _planck_t(freqs=(100.0, 143.0, 217.0, 353.0)) -> list[Tracer]:
    return [Tracer("Planck", f, spin=0) for f in freqs]


def _so_e(freqs=(90.0, 150.0)) -> list[Tracer]:
    return [Tracer("SO_SAT", f, spin=2) for f in freqs]


def _band_info(tracers: list[Tracer]) -> dict[Tracer, BandInfo]:
    return {
        t: BandInfo(tracer=t, beam_fwhm_arcmin=10.0, eff_freq_dust=_DUST_EFF[t.freq])
        for t in tracers
    }


# ---- B1. SO eff_freq_dust fallback ----


def test_build_band_info_map_so_defaults_band_freq(tiny_config):
    """SO bands without eff_freq_dust fall back to band.freq, not None."""
    pipe = Pipeline(tiny_config)
    band_map = pipe._build_band_info_map()

    so_t = Tracer("SO_SAT", 90.0, spin=2)
    assert so_t in band_map
    info = band_map[so_t]
    assert info.eff_freq_dust == 90.0
    assert info.eff_freq_cmb == 90.0

    planck_t = Tracer("Planck", 143.0, spin=2)
    info_p = band_map[planck_t]
    assert info_p.eff_freq_dust == 148.2
    assert info_p.eff_freq_cmb == 142.7


# ---- B2. TE lmin=50 floor ----


def test_tf_te_respects_lmin_floor():
    """TransferFunctionTE default lmin=50 drops bandpower bins below 50."""
    pytest.importorskip("pygsm")

    # bandpowers at lmin=20 so effective_ell spans 30..390; only bins > 50 should survive.
    bp = _bandpowers(nbins=10, bin_width=20, lmin=20, lmax=400)
    assert bp.effective_ell.min() < 50  # sanity: setup actually exercises the floor

    plk_t, plk_e, so = _planck_t(), _planck_e(), _so_e()
    band_info = {**_band_info(plk_t), **_band_info(plk_e), **_band_info(so)}
    dust = MBBDustModel(band_info=band_info)
    cmb_ref = CMBReference(
        bandpowers=bp, cls_binned={"TE": np.full(bp.effective_ell.size, 0.5)},
    )

    a_true, r_true = 1.5, 0.9
    rng = np.random.default_rng(11)

    spec_pp = Spectra(bandpowers=bp)
    for (t0, t2) in itertools.product(plk_t, plk_e):
        dust_unity = dust.predict_cross(t0, t2)
        cl = a_true * dust_unity + cmb_ref.get("TE")
        var = np.full(cl.shape, (abs(cl.mean()) * 0.02 + 1e-9) ** 2)
        cl = cl + rng.normal(0, np.sqrt(var))
        spec_pp.add(SpectrumKey(t0, t2, "TE"), cl, var)

    spec_ps = Spectra(bandpowers=bp)
    for (pt, st) in itertools.product(plk_t, so):
        dust_unity = dust.predict_cross(pt, st)
        ref = a_true * dust_unity + cmb_ref.get("TE")
        cl = r_true * ref
        var = np.full(cl.shape, (abs(cl.mean()) * 0.02 + 1e-9) ** 2)
        cl = cl + rng.normal(0, np.sqrt(var))
        spec_ps.add(SpectrumKey(pt, st, "TE"), cl, var)

    est = TransferFunctionTE(
        spec_pp_tt=spec_pp, spec_pp_te=spec_pp, spec_ps_te=spec_ps,
        cmb_ref=cmb_ref, dust=dust,
    )
    result = est.estimate(target=Tracer("SO_SAT", 90.0, spin=2))
    assert result.ell.min() > 50.0


def test_tf_te_lmin_none_uses_bp_mask():
    """lmin=None restores bp.msk behavior for users who need the full range."""
    pytest.importorskip("pygsm")

    bp = _bandpowers(nbins=10, bin_width=20, lmin=20, lmax=400)
    plk_t, plk_e, so = _planck_t(), _planck_e(), _so_e()
    band_info = {**_band_info(plk_t), **_band_info(plk_e), **_band_info(so)}
    dust = MBBDustModel(band_info=band_info)
    cmb_ref = CMBReference(
        bandpowers=bp, cls_binned={"TE": np.full(bp.effective_ell.size, 0.5)},
    )

    rng = np.random.default_rng(12)
    spec_pp = Spectra(bandpowers=bp)
    for (t0, t2) in itertools.product(plk_t, plk_e):
        cl = 1.5 * dust.predict_cross(t0, t2) + cmb_ref.get("TE")
        var = np.full(cl.shape, (abs(cl.mean()) * 0.02 + 1e-9) ** 2)
        cl = cl + rng.normal(0, np.sqrt(var))
        spec_pp.add(SpectrumKey(t0, t2, "TE"), cl, var)

    spec_ps = Spectra(bandpowers=bp)
    for (pt, st) in itertools.product(plk_t, so):
        cl = 0.9 * (1.5 * dust.predict_cross(pt, st) + cmb_ref.get("TE"))
        var = np.full(cl.shape, (abs(cl.mean()) * 0.02 + 1e-9) ** 2)
        cl = cl + rng.normal(0, np.sqrt(var))
        spec_ps.add(SpectrumKey(pt, st, "TE"), cl, var)

    est = TransferFunctionTE(
        spec_pp_tt=spec_pp, spec_pp_te=spec_pp, spec_ps_te=spec_ps,
        cmb_ref=cmb_ref, dust=dust, lmin=None,
    )
    result = est.estimate(target=Tracer("SO_SAT", 90.0, spin=2))
    assert result.ell.min() == pytest.approx(bp.effective_ell[bp.msk].min())


# ---- B3. Dust positive_only gating ----


def test_dust_fit_amplitude_te_keeps_negative():
    """positive_only=False (TE default) keeps pairs with C - CMB < 0."""
    pytest.importorskip("pygsm")

    bp = _bandpowers(nbins=5, bin_width=20, lmin=30, lmax=300)
    plk_t, plk_e = _planck_t(), _planck_e()
    band_info = {**_band_info(plk_t), **_band_info(plk_e)}
    dust = MBBDustModel(band_info=band_info)
    cmb_at = np.full(bp.effective_ell.size, 1.0)
    cmb_ref = CMBReference(bandpowers=bp, cls_binned={"EE": cmb_at, "TE": cmb_at})

    # Inject negative residual (cl - CMB = -1.5) for every pair at every ell.
    te_pairs = list(itertools.product(plk_t, plk_e))
    ee_pairs = list(itertools.combinations(plk_e, 2))
    spec = Spectra(bandpowers=bp)
    for (t1, t2) in te_pairs:
        cl = np.full(bp.effective_ell.size, -0.5)
        var = np.full(bp.effective_ell.size, 0.01)
        spec.add(SpectrumKey(t1, t2, "TE"), cl, var)
    for (t1, t2) in ee_pairs:
        cl = np.full(bp.effective_ell.size, -0.5)
        var = np.full(bp.effective_ell.size, 0.01)
        spec.add(SpectrumKey(t1, t2, "EE"), cl, var)

    # EE default (positive_only=True) → everything filtered, NaN.
    fit_ee = dust.fit_amplitude(spec, cmb_ref, ee_pairs, comp="EE", ell_idx=0)
    assert np.isnan(fit_ee.value)

    # TE default (positive_only=False) → fit runs, returns finite a_hat.
    fit_te = dust.fit_amplitude(spec, cmb_ref, te_pairs, comp="TE", ell_idx=0)
    assert np.isfinite(fit_te.value)

    # Explicit positive_only=True forces filter regardless of comp.
    fit_forced = dust.fit_amplitude(
        spec, cmb_ref, te_pairs, comp="TE", ell_idx=0, positive_only=True,
    )
    assert np.isnan(fit_forced.value)


# ---- B4. Pol-angle variance finite at a=0 ----


def test_pol_angle_variance_finite_at_zero():
    """EB signal of 0 -> a~=0; variance must remain finite via closed-form d alpha/da."""
    bp = _bandpowers(nbins=25, bin_width=20, lmin=20, lmax=520)
    so = _so_e()

    spec_ss = Spectra(bandpowers=bp)
    n = bp.effective_ell.size
    ee_var = np.full(n, 1e-4)
    bb_var = np.full(n, 1e-4)
    eb_var = np.full(n, 1e-6)
    for (t1, t2) in itertools.combinations_with_replacement(so, 2):
        ee = np.full(n, 1.0)
        bb = np.full(n, 0.1)
        eb = np.zeros(n)  # forces a_hat ~= 0
        spec_ss.add(SpectrumKey(t1, t2, "EE"), ee, ee_var)
        spec_ss.add(SpectrumKey(t1, t2, "BB"), bb, bb_var)
        spec_ss.add(SpectrumKey(t1, t2, "EB"), eb, eb_var)

    lmax_sweep = (200, 300, 500)
    est = PolarizationAngleEB(spec_ss=spec_ss, lmin=30, lmax_sweep=lmax_sweep)
    result = est.estimate()

    # With a ~= 0, dalpha/da = 0.5, so sigma_alpha = 0.5 * a_fisher.
    assert np.all(np.isfinite(result.errors))
    assert np.all(result.errors > 0)

    # Cross-check the closed-form derivative against a_fisher recomputed by hand.
    eff_ell = bp.effective_ell
    for j, lmax in enumerate(lmax_sweep):
        cap_msk = (eff_ell < lmax) & (eff_ell > 30)
        for i, (t1, t2) in enumerate(itertools.combinations_with_replacement(so, 2)):
            template = (np.full(n, 1.0) - np.full(n, 0.1))[cap_msk]
            a_fisher = fisher_error(template, eb_var[cap_msk])
            expected = 0.5 * a_fisher
            assert result.errors[i, j] == pytest.approx(expected, rel=1e-3)


# ---- C3. Reports: save_npz + plot ----


def _make_tf_result() -> "FitResult":
    from cmb_diagnostics.estimators.base import FitResult
    ell = np.linspace(40, 400, 10)
    values = np.linspace(0.3, 0.9, 10)
    errors = np.full(10, 0.02)
    return FitResult(
        name="tf_ee_SO_SAT_90",
        ell=ell, values=values, errors=errors,
        diagnostics={"r": np.sqrt(values)},
        metadata={"target": Tracer("SO_SAT", 90.0, spin=2)},
    )


def _make_pol_angle_result() -> "FitResult":
    from cmb_diagnostics.estimators.base import FitResult
    lmax_arr = np.array([200.0, 300.0, 500.0])
    n_pairs = 3
    rng = np.random.default_rng(3)
    values = rng.normal(0.0, 0.01, size=(n_pairs, lmax_arr.size))
    errors = np.full(values.shape, 0.005)
    pair_labels = np.array([[90.0, 90.0], [90.0, 150.0], [150.0, 150.0]])
    return FitResult(
        name="pol_angle_eb",
        ell=lmax_arr, values=values, errors=errors,
        diagnostics={"so_pairs": pair_labels, "lmax_sweep": lmax_arr},
        metadata={"lmin": 30},
    )


def test_tf_save_npz_roundtrip(tmp_path):
    from cmb_diagnostics.estimators.base import FitResult
    from cmb_diagnostics.reports import tf as tf_reports

    result = _make_tf_result()
    p = tmp_path / "tf.npz"
    tf_reports.save_npz(result, p)
    loaded = FitResult.load_npz(p)

    assert loaded.name == result.name
    np.testing.assert_array_equal(loaded.ell, result.ell)
    np.testing.assert_array_equal(loaded.values, result.values)
    np.testing.assert_array_equal(loaded.errors, result.errors)
    np.testing.assert_array_equal(loaded.diagnostics["r"], result.diagnostics["r"])


def test_tf_plot_returns_fig_ax_no_write(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from cmb_diagnostics.reports import tf as tf_reports

    result = _make_tf_result()
    fig, ax = tf_reports.plot([result])
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    # No files written to tmp_path.
    assert not any(tmp_path.iterdir())


def test_tf_plot_writes_when_path_given(tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    from cmb_diagnostics.reports import tf as tf_reports

    result = _make_tf_result()
    path = tmp_path / "tf_ee.png"
    tf_reports.plot([result], path=path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_pol_angle_save_npz_roundtrip(tmp_path):
    from cmb_diagnostics.estimators.base import FitResult
    from cmb_diagnostics.reports import pol_angle as pa_reports

    result = _make_pol_angle_result()
    p = tmp_path / "pa.npz"
    pa_reports.save_npz(result, p)
    loaded = FitResult.load_npz(p)

    assert loaded.name == result.name
    np.testing.assert_array_equal(loaded.values, result.values)
    np.testing.assert_array_equal(
        loaded.diagnostics["so_pairs"], result.diagnostics["so_pairs"]
    )


def test_pol_angle_plot_returns_fig_ax_no_write(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from cmb_diagnostics.reports import pol_angle as pa_reports

    result = _make_pol_angle_result()
    fig, ax = pa_reports.plot(result)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert not any(tmp_path.iterdir())


def test_pol_angle_plot_writes_when_path_given(tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    from cmb_diagnostics.reports import pol_angle as pa_reports

    result = _make_pol_angle_result()
    path = tmp_path / "pol_angle.png"
    pa_reports.plot(result, path=path)
    assert path.exists()
    assert path.stat().st_size > 0


# ---- C1. Pipeline.run end-to-end with synthetic stack ----


def _install_synthetic_pipeline_state(pipe, *, rng_seed: int = 5):
    """Monkey-patch load_mask / compute_spectra on ``pipe`` to skip real IO.

    Loads the synthetic CAMB reference (already on disk via phase3_config
    fixture) and supplies synthetic pp/ps/ss spectra that drive the fits.
    Returns the ``dust`` model installed, so callers can introspect.
    """
    from unittest.mock import MagicMock

    from cmb_diagnostics.io.camb import load_camb_reference
    from cmb_diagnostics.models.bandpowers import Bandpowers
    from cmb_diagnostics.models.dust import MBBDustModel

    def _fake_load_mask(self=pipe):
        self.mask = MagicMock()
        self.mask.fsky_effective = 0.1
        self.bandpowers = Bandpowers.from_config(self.cfg.bandpowers, self.cfg.nside)
        self.cmb_ref = load_camb_reference(self.cfg.camb, self.bandpowers, self.cfg.nside)
        return self.mask

    def _fake_build_fieldsets(self=pipe):
        self.fieldsets = {"planck": MagicMock(), "so": MagicMock()}
        return self.fieldsets

    # Build two Planck tracers (cross-pair) and two SO tracers so TF estimators
    # have non-trivial combinatorics. Bandpowers come from cfg.
    plk_e = [Tracer("Planck", 100.0, spin=2), Tracer("Planck", 143.0, spin=2)]
    plk_t = [Tracer("Planck", 100.0, spin=0), Tracer("Planck", 143.0, spin=0)]
    so = [Tracer("SO_SAT", 90.0, spin=2)]
    band_info = {
        plk_e[0]: BandInfo(plk_e[0], 9.66, eff_freq_cmb=101.31, eff_freq_dust=105.25),
        plk_e[1]: BandInfo(plk_e[1], 7.27, eff_freq_cmb=142.71, eff_freq_dust=148.235),
        plk_t[0]: BandInfo(plk_t[0], 9.66, eff_freq_cmb=101.31, eff_freq_dust=105.25),
        plk_t[1]: BandInfo(plk_t[1], 7.27, eff_freq_cmb=142.71, eff_freq_dust=148.235),
        so[0]: BandInfo(so[0], 27.4, eff_freq_cmb=90.0, eff_freq_dust=90.0),
    }
    dust = MBBDustModel(band_info=band_info)

    def _fake_compute_spectra(self=pipe):
        from cmb_diagnostics.spectra.store import Spectra

        bp = self.bandpowers
        cmb_ref = self.cmb_ref
        n = bp.effective_ell.size
        rng = np.random.default_rng(rng_seed)
        a_true, r_true = 1.5, 0.8

        # Build EE on pp (Planck auto+cross), and TE on pp (Planck T x E).
        # SS gets EE/BB/EB for pol-angle.
        pp = Spectra(bandpowers=bp)
        for (t1, t2) in itertools.product(plk_e, plk_e):
            dust_unity = dust.predict_cross(t1, t2)
            cl = a_true * dust_unity + cmb_ref.get("EE")
            var = np.full(cl.shape, (abs(cl.mean()) * 0.02 + 1e-9) ** 2)
            cl = cl + rng.normal(0, np.sqrt(var))
            pp.add(SpectrumKey(t1, t2, "EE"), cl, var)
        for (t0, t2) in itertools.product(plk_t, plk_e):
            dust_unity = dust.predict_cross(t0, t2)
            cl = a_true * dust_unity + cmb_ref.get("TE")
            var = np.full(cl.shape, (abs(cl.mean()) * 0.02 + 1e-9) ** 2)
            cl = cl + rng.normal(0, np.sqrt(var))
            pp.add(SpectrumKey(t0, t2, "TE"), cl, var)

        ps = Spectra(bandpowers=bp)
        for (pt, st) in itertools.product(plk_e, so):
            dust_unity = dust.predict_cross(pt, st)
            ref = a_true * dust_unity + cmb_ref.get("EE")
            cl = r_true * ref
            var = np.full(cl.shape, (abs(cl.mean()) * 0.02 + 1e-9) ** 2)
            cl = cl + rng.normal(0, np.sqrt(var))
            ps.add(SpectrumKey(pt, st, "EE"), cl, var)
        for (pt, st) in itertools.product(plk_t, so):
            dust_unity = dust.predict_cross(pt, st)
            ref = a_true * dust_unity + cmb_ref.get("TE")
            cl = r_true * ref
            var = np.full(cl.shape, (abs(cl.mean()) * 0.02 + 1e-9) ** 2)
            cl = cl + rng.normal(0, np.sqrt(var))
            ps.add(SpectrumKey(pt, st, "TE"), cl, var)

        ss = Spectra(bandpowers=bp)
        for (t1, t2) in itertools.combinations_with_replacement(so, 2):
            ee = np.full(n, 1.0)
            bb = np.full(n, 0.1)
            eb = np.zeros(n)
            ss.add(SpectrumKey(t1, t2, "EE"), ee, np.full(n, 1e-4))
            ss.add(SpectrumKey(t1, t2, "BB"), bb, np.full(n, 1e-4))
            ss.add(SpectrumKey(t1, t2, "EB"), eb, np.full(n, 1e-6))

        self.spectra = {"pp": pp, "ps": ps, "ss": ss}
        return self.spectra

    pipe.load_mask = _fake_load_mask
    pipe.build_fieldsets = _fake_build_fieldsets
    pipe.compute_spectra = _fake_compute_spectra
    pipe._build_dust = lambda: dust
    return dust


def test_pipeline_run_with_synthetic_stack(phase3_config, tmp_path, monkeypatch):
    pytest.importorskip("pygsm")
    import matplotlib
    matplotlib.use("Agg")

    out_dir = tmp_path / "run_out"
    # Rewrite output_dir to tmp_path so we don't pollute fixture-level out/.
    from dataclasses import replace
    cfg = replace(phase3_config, output_dir=out_dir)

    pipe = Pipeline(cfg)
    _install_synthetic_pipeline_state(pipe)

    results = pipe.run()

    assert "tf_ee_SO_SAT_90" in results
    assert "tf_te_SO_SAT_90" in results
    assert "pol_angle_eb" in results

    for name in ("tf_ee_SO_SAT_90", "tf_te_SO_SAT_90", "pol_angle_eb"):
        npz = out_dir / f"{name}.npz"
        assert npz.exists(), f"missing {npz}"
        assert npz.stat().st_size > 0

    for fname in ("tf_ee.png", "tf_te.png", "pol_angle.png"):
        p = out_dir / fname
        assert p.exists(), f"missing {p}"
        assert p.stat().st_size > 0


def test_pipeline_run_writes_diagnostic_plots(phase3_config, tmp_path):
    pytest.importorskip("pygsm")
    import matplotlib
    matplotlib.use("Agg")

    from dataclasses import replace
    adv = replace(phase3_config.advanced, write_diagnostic_plots=True)
    out_dir = tmp_path / "out2"
    cfg = replace(phase3_config, output_dir=out_dir, advanced=adv)

    pipe = Pipeline(cfg)
    _install_synthetic_pipeline_state(pipe)
    pipe.run()

    diag_dir = out_dir / "diagnostics"
    assert diag_dir.is_dir(), f"diagnostics/ not created under {out_dir}"
    diag_pngs = list(diag_dir.glob("*_diagnostics.png"))
    assert diag_pngs, f"no *_diagnostics.png in {diag_dir}"
    for p in diag_pngs:
        assert p.stat().st_size > 0


# ---- C5. CLI end-to-end with synthetic stack ----


def test_cli_tf_ee_end_to_end_synthetic(phase3_config_yaml, monkeypatch, tmp_path):
    """`cmb-diag tf-ee --config ...` writes npz+png without real IO."""
    pytest.importorskip("pygsm")
    import matplotlib
    matplotlib.use("Agg")

    # Redirect the config's output_dir to a fresh tmp location by editing the
    # YAML (quickest path — Config.from_yaml is called inside cli.main).
    cfg_text = phase3_config_yaml.read_text()
    new_out = tmp_path / "cli_out"
    cfg_text = cfg_text.replace(
        f"output_dir: {phase3_config_yaml.parent / 'out'}",
        f"output_dir: {new_out}",
    )
    new_cfg = tmp_path / "cli_cfg.yaml"
    new_cfg.write_text(cfg_text)

    from cmb_diagnostics import cli as cli_mod
    from cmb_diagnostics.pipeline import Pipeline

    original_init = Pipeline.__init__

    def patched_init(self, cfg):
        original_init(self, cfg)
        _install_synthetic_pipeline_state(self)

    monkeypatch.setattr(Pipeline, "__init__", patched_init)

    rc = cli_mod.main(["tf-ee", "--config", str(new_cfg)])
    assert rc == 0
    assert (new_out / "tf_ee_SO_SAT_90.npz").exists()
    assert (new_out / "tf_ee.png").exists()


def test_cli_tf_ee_writes_diagnostics_when_flag_set(
    phase3_config_yaml, monkeypatch, tmp_path
):
    """`cmb-diag tf-ee` must honor cfg.advanced.write_diagnostic_plots — mirrors Pipeline.run."""
    pytest.importorskip("pygsm")
    import matplotlib
    matplotlib.use("Agg")

    cfg_text = phase3_config_yaml.read_text()
    new_out = tmp_path / "cli_diag_out"
    cfg_text = cfg_text.replace(
        f"output_dir: {phase3_config_yaml.parent / 'out'}",
        f"output_dir: {new_out}",
    )
    # Append an advanced block turning on the diagnostics flag.
    cfg_text += "\nadvanced:\n  write_diagnostic_plots: true\n"
    new_cfg = tmp_path / "cli_diag_cfg.yaml"
    new_cfg.write_text(cfg_text)

    from cmb_diagnostics import cli as cli_mod
    from cmb_diagnostics.pipeline import Pipeline

    original_init = Pipeline.__init__

    def patched_init(self, cfg):
        original_init(self, cfg)
        _install_synthetic_pipeline_state(self)

    monkeypatch.setattr(Pipeline, "__init__", patched_init)

    rc = cli_mod.main(["tf-ee", "--config", str(new_cfg)])
    assert rc == 0

    diag_dir = new_out / "diagnostics"
    assert diag_dir.is_dir(), f"diagnostics/ not created under {new_out}"
    diag_pngs = list(diag_dir.glob("tf_ee_*_diagnostics.png"))
    assert diag_pngs, f"no tf_ee_*_diagnostics.png in {diag_dir}"
    for p in diag_pngs:
        assert p.stat().st_size > 0
