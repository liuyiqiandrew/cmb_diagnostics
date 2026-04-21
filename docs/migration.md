# Migration from current code

For anyone with scripts against the existing (pre-refactor) API: this document maps every current class / attribute to its new home. Both generations (V1 `PSContainer`-god-object and V2 `NmtFieldContainer`+`PSCalculator`+`SOPlkTF`) are replaced by one new architecture.

## Class-to-module map

### V1 (in `cmb_diagnoistics/`)

| V1 class / function | New location |
|---|---|
| `PSContainer` (god-object) | Split across `io.loaders`, `io.masks`, `io.camb`, `fields.builder`, `spectra.compute`, `spectra.covariance`, `models.cmb` |
| `PSContainer.init_mask` / `init_mask_from_box` | `io.masks.load(MaskConfig)` → `Mask` |
| `PSContainer.init_nmt` | `Bandpowers.from_config(BandpowersConfig)` |
| `PSContainer.init_camb_dl` | `io.camb.load_camb_reference(CambConfig, bp)` → `CMBReference` |
| `PSContainer.init_planck_f0` / `init_planck_f2` | `fields.builder.build_fieldset(InstrumentConfig, mask)` → `FieldSet` |
| `PSContainer.init_so_f0` / `init_so_f2` | same |
| `PSContainer.calc_planck_ee` / `calc_planck_te` / `calc_planck_tt_auto` / `calc_planck_ee_auto` | `spectra.compute.compute_spectra(FieldSet, FieldSet, bp, fsky)` → `Spectra` |
| `PSContainer.calc_so_ee_auto` / `calc_so_pol_specs` / `calc_so_x_planck_ee` / `calc_so_x_planck_te` | same |
| `PSContainer.calc_*_var` | folded into `compute_spectra` (returns `Spectra` with variances attached) |
| `PSContainer.planck_ee["p100xp143"]` | `spectra.get(Tracer("Planck",100,2), Tracer("Planck",143,2), "EE")` |
| `PSContainer.so_x_planck_ee["s90xp143"]` | `spectra.get(Tracer("Planck",143,2), Tracer("SO_SAT",90,2), "EE")` (tracer1 is always the NaMaster first arg — standardized) |
| `TransferFuncEE` | `estimators.TransferFunctionEE` |
| `TransferFuncEE.calc_tf_ee` | constructor takes Spectra + CMBReference + DustModel; `.estimate(target=Tracer)` returns `FitResult` |
| `TransferFuncEE.estimate_tf` | `TransferFunctionEE._fit_bin(i)` (private) |
| `TransferFuncEE.plot_and_save_transfer_func` | `reports.tf.save_npz` + `reports.tf.plot` (downstream of estimator) |
| `TransferFuncTE` | `estimators.TransferFunctionTE` |
| `PolAngEB` | `estimators.PolarizationAngleEB` |
| `PolAngEB.calc_pol_ang` | `PolarizationAngleEB(...).estimate()` |

### V2 (in `cmb_diagnoistics/Container.py`, `PSCalculator.py`, `Estimator.py`, `Models.py`)

| V2 class / function | New location |
|---|---|
| `Container.NmtFieldContainer` | `fields.FieldSet` (keyed by `Tracer` instead of bare freq; otherwise identical shape) |
| `Container.PSContainer` (typed on `PSType` enum) | `spectra.Spectra` (stores all components in one object, keyed by `SpectrumKey`) |
| `Constants.PSType` | eliminated; component is part of `SpectrumKey` as `Literal["EE","EB",...]` |
| `PSCalculator.PSCalculator` | `spectra.compute.compute_spectra(FieldSet, FieldSet, bp, fsky) → Spectra` |
| `Estimator.Fitter` | `fitting.chi2.Fitter` — constructor no longer side-effects (no auto-fit + print in `__init__`); call `.fit()` explicitly |
| `Estimator.SOPlkTF` | `estimators.TransferFunctionEE` (EE path); `__tf_te` stub replaced by real `TransferFunctionTE` |
| `Models.amp_dust_mbb` | `models.dust.MBBDustModel.predict_cross` |
| `Models.tf_model` | `models.tf.TFAmplitudeModel.predict` (trivial scalar scaling) |
| `Models.rttf_error`, `rttf_fisher` | `fitting.chi2.fisher_error` (generalized) |
| Per-bin `debug_dust_fit_*.png` / `debug_tf_fit_*.png` in CWD | `FitResult.diagnostics["dust_fit_per_bin"]` + optional `reports.tf.plot_diagnostics(result, out_dir)` — still opt-in, but writes under `cfg.output_dir` rather than CWD |

### `diag_utils.py`

| Function | New location |
|---|---|
| `box2hpmask` | `io.masks.box2hpmask` |
| `apodize_square_mask` | `io.masks.apodize_square_mask` |
| `read_carr2healpix` | `io.loaders._read_car_to_healpix` (private helper inside `SOCarLoader`) |
| `knox_covar` | `spectra.covariance.knox_variance` |
| `dust_dl` | `models.dust.MBBDustModel._dust_unit` (private, at unit amplitude) |
| `log_gauss_like`, `dust_neg_lnlike`, `tf_neg_lnlike`, `neg_amp_lnlike` | `fitting.chi2` likelihood helpers |
| `hess_inv` | `fitting.chi2.fisher_error` |
| `res2ang` | `models.tf.rotation_from_amplitude` (closed-form α = arctan(2a)/4) |
| `pol_ang_cost` | unused in current code — deleted |

### `dev/tf_calib.py`

Pre-refactor prototype. Deleted entirely (functionally subsumed by `TransferFunctionEE`).

## Usage pattern changes

### Before (V1)

```python
import cmb_diagnostics
tf = cmb_diagnostics.TransferFuncEE()
tf.planck_fname = '/custom/path/planck_{}.fits'
tf.so_fname = '/custom/path/so_f{freq:03d}.fits'
tf.camb_dl_path = '/custom/camb.dat'
tf.init_mask(mask_hp)
tf.calc_tf_ee()                  # reads maps, computes spectra, fits, plots, saves
```

### After

```python
from cmb_diagnostics import Pipeline, Config
pipe = Pipeline(Config.from_yaml("my_run.yaml"))
results = pipe.run()             # returns dict[str, FitResult]
```

Or, assembling by hand:

```python
from cmb_diagnostics import (
    Config, Bandpowers, MBBDustModel, TransferFunctionEE, Tracer,
)
from cmb_diagnostics.io import masks as mask_io
from cmb_diagnostics.io.camb import load_camb_reference
from cmb_diagnostics.fields import build_fieldset
from cmb_diagnostics.spectra import compute_spectra
from cmb_diagnostics.reports import tf as tf_reports

cfg = Config.from_yaml("my_run.yaml")
mask = mask_io.load(cfg.mask)
bp = Bandpowers.from_config(cfg.bandpowers)
cmb_ref = load_camb_reference(cfg.camb, bp)

planck = build_fieldset(cfg.planck, mask)
so     = build_fieldset(cfg.so, mask)

spec_pp = compute_spectra(planck, planck, bp, mask.fsky_effective)
spec_ps = compute_spectra(planck, so,     bp, mask.fsky_effective)

dust = MBBDustModel(cfg.dust, band_info=cfg.band_info)
tf_ee = TransferFunctionEE(spec_pp, spec_ps, cmb_ref, dust)
result = tf_ee.estimate(target=Tracer("SO_SAT", 90.0, spin=2))

tf_reports.save_npz(result, cfg.output_dir / "tf_ee_so_90.npz")
tf_reports.plot([result], cfg.output_dir / "tf_ee.png")
```

### Before (V2, `test/new_estimator_test.py`)

```python
from cmb_diagnostics.cmb_diagnoistics.Estimator import SOPlkTF
# ... manual field construction ...
TFEst = SOPlkTF(plk_pp, sp_pp, bins, lmax=600)
tf90 = TFEst.calc_tf(90)        # writes debug_dust_fit_*.png to CWD!
dtf90 = TFEst.dtf
```

### After (equivalent in new code)

```python
from cmb_diagnostics import TransferFunctionEE, Tracer
tf_ee = TransferFunctionEE(spec_pp, spec_ps, cmb_ref, dust)
result_90 = tf_ee.estimate(target=Tracer("SO_SAT", 90.0, spin=2))
# result_90.values, result_90.errors — no CWD writes
```

## What breaks (intentionally)

- Every `from cmb_diagnoistics.*` import (misspelled name) — update to `cmb_diagnostics.*`.
- Attribute-monkey-patching like `tf_ee_est.planck_fname = '...'` — move to YAML config.
- String spectrum keys like `"p100xp143"` — use `Spectra.get(Tracer, Tracer, comp)`.
- `Fitter(...)` with side effects in constructor — fit is now explicit.
- CWD side-effect writes of `debug_*_fit_*.png` — now opt-in via `reports.tf.plot_diagnostics`.
- Re-exports under `cmb_diagnostics.Container.*`, `cmb_diagnostics.PSContainer.*` (name collision with V1/V2) — gone; everything moves to properly-namespaced modules.

## Numerical divergences from V1

- **Beam length.** `fields.builder._gauss_beam` calls `hp.gauss_beam(fwhm, 3*nside - 1)` (array length `3*nside`), matching V1's `PSContainer` Planck path and the exact length required by pymaster 2.x (`ainfo.lmax + 1`). V2's `new_estimator_test.py` used `hp.gauss_beam(fwhm, 3*nside)` (one extra entry); pymaster < 2.0 silently truncated the tail, but pymaster 2.x rejects it with `RuntimeError: Passing inconsistent arguments from python` in `_nmtlib.comp_coupling_matrix`. The new code matches V1 and pymaster 2.x's contract exactly; the tail was physically inert so this does not change TF output.
- **`effective_fsky`.** Now `sum(w²)/Npix` (Gaussian/Knox convention), not V1's `sum(w)/Npix`. Knox variances differ accordingly; this is a correctness fix, not a regression.
- **Dust β.** V1's `diag_utils.dust_dl` hard-codes β = 1.53. The new `MBBDustModel.predict_cross` reads β from `cfg.dust.beta` (default 1.6 in `configs/*.yaml`, matching V2 `new_estimator_test.py`). This is the largest numerical driver of TF drift. The regression gate in `tests/integration/test_bf_tf_regression.py` compares against a golden generated by the new Pipeline at β = 1.6 (no V1-compat toggle is planned). The golden is regenerated via `scripts/regen_tf_golden.py` whenever a scientifically justified estimator change lands.
- **Fisher error form.** `cfg.advanced.fisher_error_form ∈ {"v1", "v2"}` is kept for forward compatibility but is currently a no-op. V1's `hess_inv(dmdq, var) = 1/sum(dmdq²/var)` returns the variance; V2's `rttf_error(x, sqrt(var)) = 1/sqrt(sum(x²/var))` returns the std. Because both reductions use the same `dm/dr = a·dust + cmb` and the TF error is `2|r|·√F⁻¹` in both, the resulting `FitResult.errors` are mathematically identical. The new code implements a single `fitting.chi2.fisher_error` helper (1-σ).
- **Debug PNGs.** V2's `SOPlkTF.__tf_ee` wrote `debug_dust_fit_*.png` + `debug_tf_fit_*.png` per ℓ-bin into CWD. The new estimator never touches the filesystem; per-bin fit data (`r`, `dust_amp`, `chi2_tf`) is captured in `FitResult.diagnostics`. Routing these to `reports/diagnostics/` under `cfg.output_dir` ships in Phase 5.

The `main` branch stays intact as a permanent reference for anyone who needs to reproduce a pre-refactor result.
