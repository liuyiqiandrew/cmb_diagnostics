# CMB Diagnostics

Derive calibration information (transfer functions, polarization angle) for Simons Observatory CMB maps from SO × Planck cross-spectra.

> **Status:** this branch (`refactor`) carries the new layered package at `src/cmb_diagnostics/`. `cmb-diag run --config <file>` is functional end to end (writes `.npz` + `.png` under `cfg.output_dir`). The legacy `cmb_diagnoistics/` package is gone as of Phase 6; the pre-refactor code still lives on `main` if you need to reproduce pre-refactor results.

## Why this repo

SO's map-maker filters the true sky and may rotate the polarization reference axis. Two effects have to be measured before SO maps can be used for cosmology:

1. **Transfer function** `TF(ℓ)` — the ℓ-dependent attenuation from time-domain filtering + the map-maker.
2. **Polarization angle** `α` — the rotation of SO's polarization reference axis, which leaks `EE − BB` into `EB`.

Planck is calibrated and effectively unfiltered at the relevant scales. Using SO × Planck cross-spectra breaks the degeneracy between SO's unknown TF and the true sky, and SO's own EB carries the pol-angle signal. A modified black-body dust model is fit jointly to Planck × Planck residuals and subtracted before both estimators run.

## Architecture

The refactor replaces two parallel generations of entangled code with one layered architecture:

```
 CLI / scripts / notebooks
         │
     Pipeline
     │      │
 Estimators  Reports
         │
      Models
         │
      Spectra
         │
       Fields
         │
         IO
         │
       Config
```

- **IO** — load Planck HEALPix maps, SO CAR maps, masks, beams, CAMB reference.
- **Fields** — build NaMaster `NmtField` objects keyed by typed `Tracer`.
- **Spectra** — compute all required Cℓ and per-bin (Knox) variances in one pass.
- **Models** — CMB reference, MBB dust, TF amplitude, rotation.
- **Estimators** — `TransferFunctionEE`, `TransferFunctionTE`, `PolarizationAngleEB`. Stateless after construction; return `FitResult`.
- **Reports** — plotting + `.npz` output. No side-effects inside estimators.
- **Pipeline / CLI** — YAML config → end-to-end run.

Full details:

- [`docs/architecture.md`](docs/architecture.md) — layering, module index, dependency rules.
- [`docs/data_model.md`](docs/data_model.md) — canonical objects (`Tracer`, `FieldSet`, `Spectra`, `FitResult`, ...).
- [`docs/api.md`](docs/api.md) — public API, typical usage snippets.
- [`docs/science_reference.md`](docs/science_reference.md) — every formula (TF, α, dust MBB, Knox) with source pointers.
- [`docs/configuration.md`](docs/configuration.md) — YAML schema.
- [`docs/migration.md`](docs/migration.md) — mapping from current classes/attributes to the new ones.

## Planned usage (after implementation phases land)

```python
from cmb_diagnostics import Pipeline, Config

pipe = Pipeline(Config.from_yaml("configs/satp3_south.yaml"))
results = pipe.run()                     # dict[str, FitResult]
results["tf_ee_so_90"].save_npz("tf_ee_so_90.npz")
```

Or via CLI (after `pip install -e .`):

```
cmb-diag run --config configs/satp3_south.yaml
```

Example configs in `configs/`:

- [`configs/satp3_south.yaml`](configs/satp3_south.yaml) — SATp3, box-mask south + east patches (reproduces `test/example.py` setup).
- [`configs/iso_satp3.yaml`](configs/iso_satp3.yaml) — SATp3 with the ISO analysis mask (reproduces `test/new_container_test.py` setup).

## Dependencies

- `pymaster` (NaMaster)
- `healpy`, `pixell` (HEALPix and CAR pixelization)
- `scipy`, `numpy`, `matplotlib`
- `pygsm` — **the fork at `github.com/liuyiqiandrew/pygsm`**, not the PyPI package. Provides `trj2tcmb` and `planck_law`.

### Install (Phase 2+)

```
pip install -e ".[dev]"
pip install git+https://github.com/liuyiqiandrew/pygsm.git
```

The `pygsm` fork is a separate step because it isn't on PyPI and declaring a VCS
dependency in `pyproject.toml` would break installs for anyone without git + SSH.

### Contributor note: `PYTHONPATH` footgun

The package lives at `src/cmb_diagnostics/` and is reached after
`pip install -e ".[dev]"`. **Unset any `PYTHONPATH` entries pointing at this
repo's parent directory** before installing — otherwise Python may resolve
`import cmb_diagnostics` to a stale checkout elsewhere on disk. The
session-scoped guard in `tests/conftest.py` catches this and fails the suite
with a clear message if `cmb_diagnostics` does not resolve to `src/cmb_diagnostics/`.

### Notebooks

See [`notebooks/`](notebooks/) for jupytext-format example scripts covering the
one-call, step-through, and inline-dict calling patterns. Install notebook tooling
with the `notebook` extra: `pip install -e ".[dev,notebook]"`.

## Refactor roadmap

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Design docs (`docs/`) + example configs + refactor README | **done** (this branch) |
| 2 | Package scaffold + `pyproject.toml` + pytest fixtures (no functional code) | **done** |
| 3 | Port `io/`, `fields/`, `spectra/`, `models/` + component unit tests | **done** |
| 4 | Port estimators; implement `TransferFunctionTE` properly; add end-to-end regression test against `test/bf_tf.npy` / `ml_tf.npy` | **done** |
| 5 | Pipeline + CLI + reports; `cmb-diag run` is functional end to end | **done** |
| 6 | Delete `cmb_diagnoistics/`, `dev/tf_calib.py`, `result/`; wire diagnostic plots to `cfg.output_dir` | **done** |

The pre-refactor `cmb_diagnoistics/` god-object code is preserved on the `main`
branch if you need to reproduce pre-refactor results.

## Licence

See `licence` at repo root.
