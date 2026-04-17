# Notebook examples

These scripts are written in [jupytext](https://jupytext.readthedocs.io/) "percent" format
(`# %%` cell markers) so they can be run as plain `.py` files **and** converted to
`.ipynb` on the fly:

```
jupytext --to ipynb notebooks/01_quickstart.py
jupyter lab notebooks/01_quickstart.ipynb
```

The `.py` versions are tracked in git; `.ipynb` conversions are gitignored to avoid
large JSON diffs.

## Three calling patterns

**A — one-call end-to-end.** See `01_quickstart.py`.

```python
pipe = Pipeline(Config.from_yaml("configs/satp3_south.yaml"))
results = pipe.run()
```

**B — step-through with inspection.** See `02_step_through.py`. Each stage populates a
`Pipeline` attribute (`pipe.mask`, `pipe.fieldsets`, `pipe.spectra`, `pipe.results`) so
intermediates can be inspected between calls — useful in a notebook for debugging a
single stage without rerunning the whole thing.

**C — inline dict config.** Skip the YAML file entirely:

```python
cfg = Config.from_dict({"schema_version": "1", "nside": 256, ...})
pipe = Pipeline(cfg)
```

## Status (Phase 2)

The scaffold is in place but estimator bodies are not yet ported. Both notebooks will
import cleanly and run up to the first estimator call, then raise `NotImplementedError`
with a pointer to the phase that fills in the body. Phase 3 makes `load_mask`,
`build_fieldsets`, and `compute_spectra` real; Phase 4 makes the estimators real;
Phase 5 makes `.run()` and plotting real.
