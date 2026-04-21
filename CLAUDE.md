# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Derive calibration information (transfer functions, polarization angle) for Simons Observatory (SO) CMB maps from SO × Planck cross-spectra. Written against Della (Princeton HPC) paths; the default constants in the code point at user `yl9946`'s data locations.

## Installation / import

There is no `setup.py` or `pyproject.toml`. The repo is used by adding its **parent directory** to `PYTHONPATH` and `import cmb_diagnostics`.

Directory vs. import name: the outer package directory is `cmb_diagnostics/` (what users import), but the inner Python package it wraps is misspelled as `cmb_diagnoistics/` (with an extra `i`). The outer `__init__.py` does `from .cmb_diagnoistics import *`, so public API appears under `cmb_diagnostics.*`. Inside the package, all intra-package imports use the misspelled `cmb_diagnoistics` name — preserve this spelling when editing.

## External dependencies (non-obvious)

- **pygsm** — must be the fork `https://github.com/liuyiqiandrew/pygsm` (not PyPI pygsm). Used for `trj2tcmb` and `planck_law`.
- **namaster** (imported as `pymaster`) — all spectrum computation goes through `nmt.compute_full_master`.
- **cmb_utils** (optional, `https://github.com/liuyiqiandrew/cmb_utils`) — some tests pull `carrbox2hpmask` / `apodize_square_mask` from it; equivalents exist in `diag_utils.py`.
- healpy, pixell (for CAR↔HEALPix reprojection), scipy, numpy, matplotlib.

## Running things

No test framework. Each file in `test/` is a standalone script with a `main()`; run `python test/<name>.py` from the `test/` directory (most scripts write output images/`.npy` to CWD). Expect multi-minute runtimes — they read full-sky Planck maps and run NaMaster MASTER.

- `test/example.py` — legacy `PSContainer`-based API end-to-end (EE TF, TE TF, pol angle).
- `test/new_container_test.py` — newer `NmtFieldContainer` + `PSCalculator` pipeline, plots Planck auto and SO×Planck EE.
- `test/new_estimator_test.py` — newer pipeline feeding `SOPlkTF`; saves `bf_tf.npy` or `ml_tf.npy` (swap by uncommenting the `so_fnames` line for FB vs. ML maps).
- `test/test_fitter.py` — standalone sanity check for `Estimator.Fitter`.
- `test/tf_plotter.py` — loads `bf_tf.npy` + `ml_tf.npy` and makes the comparison plot.

## Architecture — two coexisting generations

The repo contains **two generations** of the same pipeline. New work should go into the V2 API; V1 is kept because `test/example.py` and the README still describe it.

### V1 — monolithic, PSContainer-rooted

`cmb_diagnoistics/PSContainer.py` is a god-object that holds: hard-coded frequencies (Planck 100/143/217/353, SO 90/150), beam FWHMs, file-name templates, NaMaster bins, mask, CAMB reference spectra, and dictionaries of all computed Planck/SO/cross spectra keyed by strings like `"p100xp143"`, `"s90xp150"`, `"p100xs90"`.

Estimators inherit from it and add only the fitting step:
- `TransferFuncEstimator.py::TransferFuncEE.calc_tf_ee()` — orchestrates full pipeline: init mask → NaMaster bins → CAMB → Planck spin-2 fields → Planck EE auto+cross → Knox variance → SO spin-2 fields → SO auto → SO×Planck EE → variance → per-ell dust amplitude fit (Planck-only) → per-ell TF fit (SO×Planck).
- `TransferFuncEstimator.py::TransferFuncTE.calc_tf_te()` — same shape but uses T×E spectra (Planck T, SO E), with `lmin=50`.
- `PolAngEstimator.py::PolAngEB.calc_pol_ang()` — fits polarization angle from SO EB using `EE − BB` template.

Results (`.npy` + `.png`) are saved under `../result/` relative to the package, resolved in `PSContainer.__init__`.

To override the hard-coded data paths, monkey-patch the attributes *after* constructing the estimator (pattern shown in the README): e.g. `est.planck_fname = '...'`, `est.so_fname = '...'`, `est.camb_dl_path = '...'`. Don't edit `PSContainer.py` defaults.

### V2 — decoupled containers + calculators + estimators

- `Container.py::NmtFieldContainer` — holds `f0s` (spin-0) and `f2s` (spin-2) `nmt.NmtField` objects, keyed by arbitrary hashables (typically frequency).
- `Container.py::PSContainer` **(different class, same name as V1!)** — holds computed power spectra keyed by `(freq1, freq2)` tuples, tagged with a `PSType` enum (`PP`/`TP`/`TT`) from `Constants.py`. This namespace collision is real: `cmb_diagnoistics.PSContainer` (module) exports the V1 class, while `cmb_diagnoistics.Container.PSContainer` is V2. Imports like `from .Container import PSContainer` in `Estimator.py` resolve to V2. Don't confuse them.
- `PSCalculator.py::PSCalculator` — given two `NmtFieldContainer`s, runs `nmt.compute_full_master` for all tracer pairs and fills a `PSContainer` (V2) with spectra **and** per-bin Knox-formula variance. Auto-computes the per-tracer auxiliary autos needed for Knox.
- `Models.py` — shared parametric models: `amp_dust_mbb` (modified black-body with fitted amplitude), `tf_model` (scalar scaling), plus Fisher-based TF error helpers (`rttf_fisher`, `rttf_error`).
- `Estimator.py::Fitter` — thin wrapper around `scipy.optimize.minimize` for χ² fits with optional `dy`; returns a callable `fitted_fun`.
- `Estimator.py::SOPlkTF` — V2 transfer-function estimator. Consumes a Planck×Planck `PSContainer` and a Planck×SO `PSContainer`. Per ℓ-bin: fits dust amplitude to (Planck×Planck − CMB) using `amp_dust_mbb`, builds the Planck-side model prediction at SO-side frequencies, fits a scalar TF against SO×Planck, stores `self.tf` (squared) and `self.dtf` (delta transferred via `rttf_error`). `PSType.TP` branch (`__tf_te`) is a stub.

Common utilities live in `diag_utils.py`: `knox_covar`, mask construction (`box2hpmask`, `apodize_square_mask`), CAR→HEALPix (`read_carr2healpix`), dust model (`dust_dl`), negative-log-likelihood helpers.

## Pitfalls / quirks

- **Units**: raw Planck/SO maps are read and multiplied by `1e6` (K → μK). All spectra are in μK²; don't re-scale.
- **Coordinate & pixelization conversions**: Planck maps must be equatorial HEALPix; SO maps are CAR and get reprojected via `read_carr2healpix`. All maps are `ud_grade`'d to `nside=512`.
- **CAMB file format**: `init_camb_dl` / `SOPlkTF.__init__` read the BBPower-style `camb_lens_nobb.dat`, prepend a zero row (to shift from `ell=2` to `ell=0`), and take columns 2 (EE), 3 (BB), 4 (TE) as Dℓ before dividing by `ell(ell+1)/(2π)` to get Cℓ.
- **Binning is `is_Dell=True`** in `nmt.NmtBin`; `self.e_dl2cl = 2π / ell / (ell+1)` is applied so downstream values are Cℓ.
- **`SOPlkTF.__tf_ee` writes `debug_dust_fit_*.png` and `debug_tf_fit_*.png` to CWD** (one per ℓ-bin). This is intentional debug output, not a bug — but it produces many files per run. Already tracked in git (many untracked debug PNGs in `test/`).
- **`__pycache__` is committed** in the repo (visible in `git status`). Don't feel the need to clean these; they're part of the tracked state.
- Key naming is asymmetric: Planck×Planck uses `pAxpB` (A < B), Planck×SO TE uses `pFpxsFs` (Planck first), but SO×Planck EE uses `sFsxpFp` (SO first). When adding spectra look at how the consuming code indexes them rather than guessing the order.
- `TransferFuncEstimator.TransferFuncEE.__generate_freq_data` has inconsistent loop indices (the `for i, ((f1,f2),(fd1,fd2)) in enumerate(...)` block overwrites the Planck rows starting at index `i`, but indexing is fragile — any change to `planck_freqs` length breaks the hard-coded slice `6:10`, `10:` etc. in `estimate_tf`). Both V1 estimators assume exactly 4 Planck freqs and 2 SO freqs.

## Where legacy prototype lives

`dev/tf_calib.py` is a predecessor of `TransferFuncEE` from before the `PSContainer` split. It's not imported by the package (the `from ..dev.tf_calib import *` line in the package `__init__.py` is commented out). Don't modify it — treat as history.
