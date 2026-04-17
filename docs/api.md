# Public API

Everything documented here is imported directly from the top-level package:

```python
from cmb_diagnostics import (
    Config, Pipeline,
    Tracer, BandInfo,
    Mask, Bandpowers, FieldSet, Spectra, SpectrumKey,
    CMBReference, MBBDustModel,
    TransferFunctionEE, TransferFunctionTE, PolarizationAngleEB,
    FitResult,
)
```

Modules below the top level (`cmb_diagnostics.io.*`, `cmb_diagnostics.spectra.*`, etc.) are usable but not guaranteed stable.

## High-level surface

### `Pipeline`

One-call entry point for end users.

```python
pipe = Pipeline(Config.from_yaml("configs/satp3_south.yaml"))
results = pipe.run()            # dict[str, FitResult]
results["tf_ee_so_90"].save_npz(...)
```

Internally orchestrates IO → fields → spectra → models → estimators → reports.

### `Config`

Typed configuration loaded from YAML. See `configuration.md` for the full schema.

```python
cfg = Config.from_yaml(path)
cfg.output_dir            # Path
cfg.bandpowers            # BandpowersConfig (bin_width, lmin, lmax, is_Dell)
cfg.mask                  # MaskConfig (file or boxes)
cfg.camb                  # CambConfig (path, format)
cfg.dust                  # DustConfig (beta, Td, nu0)
cfg.planck                # InstrumentConfig (bands, map_template, ...)
cfg.so                    # InstrumentConfig
cfg.pol_angle             # PolAngleConfig (fit_ranges)
```

`Config.from_dict(dict)` also exists for tests / programmatic construction.

## Mid-level surface (used when you want to assemble the pipeline by hand)

### Building fields

```python
from cmb_diagnostics.fields import build_fieldset
from cmb_diagnostics.io import masks as mask_io

mask = mask_io.load(cfg.mask)                            # -> Mask
planck_fields = build_fieldset(cfg.planck, mask)         # -> FieldSet
so_fields     = build_fieldset(cfg.so, mask)
```

### Computing spectra

```python
from cmb_diagnostics.spectra import compute_spectra

bp       = Bandpowers.from_config(cfg.bandpowers)
spec_pp  = compute_spectra(planck_fields, planck_fields, bp, mask.fsky_effective)
spec_ps  = compute_spectra(planck_fields, so_fields,     bp, mask.fsky_effective)
spec_ss  = compute_spectra(so_fields,     so_fields,     bp, mask.fsky_effective)

cl_ee, var_ee = spec_pp.get(t_p100, t_p143, comp="EE")
```

### Models

```python
from cmb_diagnostics.io.camb import load_camb_reference
from cmb_diagnostics import MBBDustModel

cmb_ref = load_camb_reference(cfg.camb, bp)              # -> CMBReference
dust    = MBBDustModel(beta=cfg.dust.beta, Td=cfg.dust.Td,
                       nu0=cfg.dust.nu0, band_info=cfg.band_info)
```

### Estimators

Each estimator is stateless-after-construction: pass data in, call `.estimate()`, receive `FitResult`.

```python
tf_ee = TransferFunctionEE(spec_pp, spec_ps, cmb_ref, dust)
result_90  = tf_ee.estimate(target=Tracer("SO_SAT", 90.0, spin=2))
result_150 = tf_ee.estimate(target=Tracer("SO_SAT", 150.0, spin=2))

tf_te = TransferFunctionTE(spec_pp_tt, spec_pp_te, spec_ps_te, cmb_ref, dust)
result_te_90 = tf_te.estimate(target=Tracer("SO_SAT", 90.0, spin=2))

pa = PolarizationAngleEB(spec_ss, ell_ranges=cfg.pol_angle.fit_ranges)
pa_result = pa.estimate()                    # sweeps lmax cap, returns array of results
```

### Reports

```python
from cmb_diagnostics import reports

reports.tf.save_npz(result_90,  cfg.output_dir / "tf_ee_so_90.npz")
reports.tf.plot([result_90, result_150], cfg.output_dir / "tf_ee.png",
                reference_tf=Path("/scratch/gpfs/yl9946/iso_maps/TF/…"))
reports.pol_angle.save_npz(pa_result, cfg.output_dir / "pol_angle.npz")
reports.pol_angle.plot(pa_result, cfg.output_dir / "pol_angle.png")
```

Plotting + saving never happens inside estimators. Reports take `FitResult` only.

## CLI

Installed via `pyproject.toml` entry points after `pip install -e .`:

```
cmb-diag tf-ee      --config configs/satp3_south.yaml
cmb-diag tf-te      --config configs/satp3_south.yaml
cmb-diag pol-angle  --config configs/satp3_south.yaml
cmb-diag run        --config configs/satp3_south.yaml   # all three
```

Each command is a thin wrapper: parse config → construct `Pipeline` → call the relevant estimator(s) → save reports to `cfg.output_dir`.

## Stability guarantees

- Top-level imports from `cmb_diagnostics` are the public API. Breaking changes get a major version bump.
- Anything under `cmb_diagnostics._internal` or prefixed with `_` is private.
- `io`, `spectra`, `models`, `estimators`, `fields`, `fitting`, `reports` submodules are usable; their top-level helpers are stable but exotic internals are not.
- YAML config schema is versioned. Breaking changes require `schema_version:` bump in the YAML + documented migration.
