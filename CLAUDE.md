# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Derive calibration information (transfer functions, polarization angle) for Simons Observatory (SO) CMB maps from SO × Planck cross-spectra. Written against Della (Princeton HPC) paths; default constants in example configs under `configs/` point at user `yl9946`'s data locations.

## Installation / import

Editable install via `pyproject.toml`:

```
pip install -e ".[dev]"
pip install git+https://github.com/liuyiqiandrew/pygsm.git
```

The `pygsm` fork is a separate step because it isn't on PyPI.

**PYTHONPATH footgun:** unset any `PYTHONPATH` pointing at this repo's parent directory before installing — otherwise Python can resolve `import cmb_diagnostics` to a stale checkout elsewhere. The session-scoped guard in `tests/conftest.py` catches this and fails the suite with a clear message if `cmb_diagnostics` doesn't resolve to `src/cmb_diagnostics/`.

The pre-refactor misspelled package (`cmb_diagnoistics/`) is gone as of Phase 6; it still lives on `main` if you need to reproduce pre-refactor results.

## External dependencies (non-obvious)

- **pygsm** — must be the fork `https://github.com/liuyiqiandrew/pygsm` (not PyPI pygsm). Used for `trj2tcmb` and `planck_law`.
- **namaster** (imported as `pymaster`) — all spectrum computation goes through `nmt.compute_full_master`.
- healpy, pixell (for CAR↔HEALPix reprojection), scipy, numpy, matplotlib.

## Running things

Tests use pytest; `testpaths = ["tests"]` in `pyproject.toml`.

- `pytest tests/unit` — fast unit tests; covers io, spectra, models, fitting, estimators, reports, pipeline.
- `pytest tests/integration` — end-to-end regression vs. the frozen V2 goldens at `test/bf_tf.npy` / `test/ml_tf.npy`; marked slow and requires full Planck/SO maps on disk.
- `cmb-diag run --config configs/<file>.yaml` — end-to-end CLI. Writes `{FitResult.name}.npz` and combined `tf_ee.png` / `tf_te.png` / `pol_angle.png` under `cfg.output_dir`.
- `notebooks/01_quickstart.py`, `notebooks/02_step_through.py` — jupytext scripts showing one-call and step-through calling patterns.

## Architecture — layered package

Dependencies flow strictly downward. Each module owns one concern; no god-objects.

```
 CLI / scripts / notebooks
         │
     Pipeline                     (Config → {FitResult})
     │      │
 Estimators  Reports              (Config + Spectra + Models → FitResult; plotting + .npz)
         │
      Models                       (CMB, dust MBB, TF amplitude, rotation)
         │
      Spectra                      (compute, Knox covariance, store)
         │
       Fields                      (NaMaster NmtField builder, typed Tracer)
         │
         IO                        (Planck HEALPix loader, SO CAR loader, masks, CAMB reference)
         │
       Config                      (YAML → typed dataclass)
```

Key public objects (re-exported from `cmb_diagnostics.__init__`):

- `Config` / `Config.from_yaml` / `Config.from_dict` — YAML or inline-dict construction.
- `Pipeline` — `pipe.run()` or step-by-step (`load_mask`, `build_fieldsets`, `compute_spectra`, `estimate_tf_ee`, `estimate_tf_te`, `estimate_pol_angle`). Step methods populate attributes so notebook users can inspect every intermediate.
- `Tracer(instrument, freq, spin)`, `SpectrumKey` — typed identifiers; no more string keys like `"s90xp150"`.
- `FieldSet`, `Spectra`, `FitResult` — containers (see `docs/data_model.md`).
- `TransferFunctionEE`, `TransferFunctionTE`, `PolarizationAngleEB` — stateless after construction; call `.estimate(target=Tracer(...))` → `FitResult`.
- `reports.tf.save_npz` / `reports.tf.plot` / `reports.tf.plot_diagnostics` / `reports.pol_angle.*` — downstream plotting and IO.

Full reference: `docs/architecture.md`, `docs/data_model.md`, `docs/api.md`, `docs/configuration.md`, `docs/science_reference.md`, `docs/migration.md`.

## Pitfalls / quirks (still applicable after refactor)

- **Units**: raw Planck/SO maps are multiplied by `1e6` (K → μK) at load time. All spectra are in μK²; don't re-scale.
- **Coordinate & pixelization**: Planck maps must be equatorial HEALPix; SO maps are CAR and get reprojected via `io.loaders.read_carr2healpix`. Maps are `ud_grade`'d to `cfg.nside` (default 512).
- **CAMB file format**: `io.camb.load_camb_reference` reads BBPower-style `camb_lens_nobb.dat`, prepends a zero row (to shift from `ell=1` to `ell=0`), takes columns 2 (EE), 3 (BB), 4 (TE) as Dℓ, then divides by `ell(ell+1)/(2π)` to get Cℓ.
- **Binning is `is_Dell=True`** in `nmt.NmtBin`; `Bandpowers.e_dl2cl = 2π / ell / (ell+1)` is applied so downstream values are Cℓ.
- **Beam array length**: `fields.builder._gauss_beam` calls `hp.gauss_beam(fwhm, 3*nside - 1)` (length `3*nside`). pymaster 2.x is strict about this (`ainfo.lmax + 1`); don't pass `3*nside` by accident.
- **Effective fsky**: `effective_fsky(w) = sum(w²)/Npix` (Knox convention), not `sum(w)/Npix`. Knox variances on TF / pol-angle errors depend on this.
- **Dust β default is 1.6** in example configs; V1's hard-coded `1.53` is not preserved. The integration regression golden was generated at β = 1.6.
- **Debug diagnostics**: the legacy code wrote `debug_*_fit_*.png` per ℓ-bin into CWD. The new pipeline captures per-bin fit data (`r`, `dust_amp`, `chi2_tf`) in `FitResult.diagnostics` and routes optional plots to `cfg.output_dir/diagnostics/` when `cfg.advanced.write_diagnostic_plots=True`.
