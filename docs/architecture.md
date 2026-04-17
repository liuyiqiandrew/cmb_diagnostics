# Architecture

## Goal of the refactor

The `cmb_diagnostics` codebase performs a narrow, self-contained scientific task: derive SO map-maker transfer functions and polarization-angle miscalibration from SO × Planck cross-spectra. The current implementation is scientifically correct but structurally entangled — there are two parallel generations (`PSContainer`-god-object and `NmtFieldContainer`+`PSCalculator`+`SOPlkTF`), hard-coded per-user paths, a directory-name typo that leaks into every import, and plotting/IO side-effects baked into fitting code. This refactor replaces both generations with a single layered architecture without altering any science.

## Design principles

1. **Layering — dependencies flow downward.** Each layer only imports from layers strictly below it. No cyclic or lateral hidden coupling.
2. **No god-objects.** Each class owns one concern: IO, field construction, spectrum computation, modeling, fitting, or reporting. Composition, not inheritance.
3. **Typed identifiers over strings.** `Tracer`, `SpectrumKey` replace string concatenation like `"s90xp150"`.
4. **Configuration over monkey-patching.** A `Config` object (YAML + dataclass) captures paths, frequencies, beams, fit ranges. No `est.planck_fname = ...` patterns.
5. **Reports are downstream.** Plotting, `.png` / `.npy` writes happen in a `reports/` module invoked *after* estimation. No plotting inside fitters.
6. **Side-effect-free constructors.** Constructors set state only. Execution is triggered by explicit `.run()` / `.estimate()` calls.
7. **The science is copied verbatim.** Knox, MBB, TF=√amp², α=arctan(2a)/4 keep their exact current forms. Regression tests pin them.

## Layering diagram

```
+------------------------------------------+
|   CLI / scripts / notebooks              |   <- entry points
+------------------------------------------+
|   Pipeline                               |   <- Config → FitResult
+------------------------------------------+
|   Estimators          Reports            |   <- Config + Spectra + Models → FitResult
+------------------------------------------+
|   Models                                 |   <- CMB, dust, TF, rotation
+------------------------------------------+
|   Spectra (compute / covariance / store) |   <- FieldSet pair → Spectra
+------------------------------------------+
|   Fields                                 |   <- Maps + beams + mask → NmtField objects
+------------------------------------------+
|   IO (loaders / masks / CAMB)            |   <- files → in-memory arrays
+------------------------------------------+
|   Config                                 |   <- YAML → typed dataclass
+------------------------------------------+
|   fitting/     (χ² wrapper, Fisher errors — shared utility, below models/estimators)
+------------------------------------------+
```

### Dependency rules

- `config` depends on nothing internal.
- `io` depends on `config` only.
- `fields` depends on `io` and `config`.
- `spectra` depends on `fields` and `config`.
- `models` depends on `config` and shared types (e.g. `Tracer`).
- `fitting` depends on nothing internal; pure math helper.
- `estimators` depends on `spectra`, `models`, `fitting`.
- `reports` depends on `estimators`' `FitResult` and standard plotting libs.
- `pipeline` depends on everything except CLI.
- `cli` depends on `pipeline` and `config`.

Never the reverse, and never skip layers (e.g., `estimators` shouldn't directly read files).

## Module index

| Path | Responsibility | Key exports |
|---|---|---|
| `cmb_diagnostics/__init__.py` | Public API re-exports. | `Config, Pipeline, Tracer, FieldSet, Spectra, ...` |
| `cmb_diagnostics/config.py` | Typed config + YAML loader. | `Config.from_yaml`, sub-configs |
| `cmb_diagnostics/pipeline.py` | End-to-end orchestration. | `Pipeline.run()` |
| `cmb_diagnostics/io/loaders.py` | Pluggable map loaders. | `PlanckHealpixLoader`, `SOCarLoader`, `MapLoader` Protocol |
| `cmb_diagnostics/io/masks.py` | Mask construction + apodization + effective fsky. | `apodize_square_mask`, `box2hpmask`, `effective_fsky` |
| `cmb_diagnostics/io/camb.py` | Parse BBPower-style CAMB tables → binned CMB reference. | `load_camb_reference` |
| `cmb_diagnostics/fields/builder.py` | Build NaMaster fields from raw maps + beams + mask. | `build_spin0_field`, `build_spin2_field`, `build_fieldset` |
| `cmb_diagnostics/spectra/compute.py` | Run MASTER over tracer pairs. | `compute_spectra(fieldset_a, fieldset_b, bandpowers, fsky)` |
| `cmb_diagnostics/spectra/covariance.py` | Knox variance (and hooks for future proper covariance). | `knox_variance` |
| `cmb_diagnostics/spectra/store.py` | `Spectra` container + save/load. | `Spectra`, `SpectrumKey` |
| `cmb_diagnostics/models/dust.py` | Modified black-body dust amplitude model. | `MBBDustModel` |
| `cmb_diagnostics/models/cmb.py` | Binned CMB reference container. | `CMBReference` |
| `cmb_diagnostics/models/tf.py` | Scalar-TF model, pol-angle rotation model. | `TFAmplitudeModel`, `RotationModel` |
| `cmb_diagnostics/estimators/base.py` | `Estimator` protocol + `FitResult`. | — |
| `cmb_diagnostics/estimators/transfer_function.py` | TF EE, TF TE estimators. | `TransferFunctionEE`, `TransferFunctionTE` |
| `cmb_diagnostics/estimators/pol_angle.py` | EB → polarization angle. | `PolarizationAngleEB` |
| `cmb_diagnostics/fitting/chi2.py` | χ² minimizer + Fisher error helpers. | `Fitter`, `fisher_error` |
| `cmb_diagnostics/reports/tf.py` | Plot + save TF results. | `save_tf_npz`, `plot_tf` |
| `cmb_diagnostics/reports/pol_angle.py` | Plot + save pol-angle results. | `save_pa_npz`, `plot_pa` |
| `cmb_diagnostics/cli.py` | Console entry points. | `cmb-diag tf-ee / tf-te / pol-angle / run` |

## Side-by-side deletion list (final state)

These live in the old tree and will be removed once Phase 6 lands:
- `cmb_diagnoistics/` (misspelled inner package): deleted.
- `__init__.py` at repo root (outer wrapper): deleted.
- `dev/tf_calib.py`: deleted (prototype superseded).
- Tracked `__pycache__/`: removed and added to `.gitignore`.
- `result/`: replaced by configurable `output_dir`. Existing outputs remain on `main` for reference.

## What stays the same

- All scientific formulas and their numeric values (MBB constants, CAMB file handling, Knox formula, TF squaring, α from EB fit).
- The `pygsm` fork dependency.
- The default frequency sets (Planck 100/143/217/353, SO 90/150) — but no longer hard-coded into slice offsets; they live in YAML.
- nside=512 default, Δℓ=20 default, fit range 30 ≤ ℓ ≤ 300 default — all overridable via config.

## What becomes possible after the refactor

- Adding a new instrument (e.g. SO LAT, ACT) = new entry in config + optional new `MapLoader`. No changes to estimators.
- Adding a new foreground model (synchrotron, two-component dust) = implement the `DustModel`-like protocol; drop into `TransferFunctionEE`'s constructor.
- Running on a second user's account = change paths in YAML. No code edits.
- Regression-testing a change = pytest fixtures compare new-code outputs against frozen baselines from current code.
- Finishing the TE transfer function = implement one class; the rest of the infrastructure is ready.
