# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Step-through (Pattern B) — inspect intermediates at each stage
#
# Same pipeline as `01_quickstart.py`, but each stage is called separately so you can
# peek at `pipe.mask`, `pipe.fieldsets`, `pipe.spectra`, and per-bin diagnostics on
# the returned `FitResult`.
#
# Until Phase 3 lands the step calls raise `NotImplementedError`; until Phase 4 the
# estimator calls raise. The attribute surface is stable now so Phase 3/4 contributors
# can wire real bodies without reshaping the notebook.

# %%
from cmb_diagnostics import Config, Pipeline, Tracer

cfg = Config.from_yaml("../configs/satp3_south.yaml")
pipe = Pipeline(cfg)

# %% [markdown]
# ## Stage 1 — mask
#
# `pipe.mask` is `None` before this call and a `Mask` after.

# %%
pipe.load_mask()
pipe.mask

# %% [markdown]
# ## Stage 2 — fieldsets
#
# One `FieldSet` per instrument. `pipe.fieldsets` is a dict keyed by `"planck"` / `"so"`.

# %%
pipe.build_fieldsets()
pipe.fieldsets

# %% [markdown]
# ## Stage 3 — spectra
#
# All Cℓ + Knox variances in one pass. `pipe.spectra` is a dict keyed by what kind of
# pair was computed (`"pp"`, `"ps"`, `"ss"`).

# %%
pipe.compute_spectra()
pipe.spectra

# %% [markdown]
# ## Stage 4 — estimate, inspect, plot
#
# Each estimator returns a `FitResult` that you can peek at without writing anything
# to disk. `diagnostics` holds per-bin intermediates (e.g. the fitted dust amplitude).

# %%
tf90 = pipe.estimate_tf_ee(target=Tracer("SO_SAT", 90.0, spin=2))
tf90

# %%
tf90.diagnostics.keys()

# %%
fig, ax = tf90.plot()
fig
