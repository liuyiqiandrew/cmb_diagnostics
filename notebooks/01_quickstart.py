# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Quickstart (Pattern A) — one-call end-to-end
#
# Runs the whole pipeline from a YAML config and renders the resulting `FitResult`
# objects inline. Until Phase 5 lands, `pipe.run()` raises `NotImplementedError`.

# %%
from cmb_diagnostics import Config, Pipeline, Tracer

# %% [markdown]
# ## 1. Load a config
#
# Either from a YAML file (shown) or inline via `Config.from_dict({...})`.

# %%
cfg = Config.from_yaml("../configs/satp3_south.yaml")
cfg

# %% [markdown]
# ## 2. Construct the pipeline
#
# `Pipeline` is the single object shared between the CLI and notebook entry points.
# At construction time it only stores `cfg`; nothing heavy runs yet.

# %%
pipe = Pipeline(cfg)
print("mask:", pipe.mask)
print("fieldsets:", pipe.fieldsets)
print("spectra:", pipe.spectra)
print("results:", pipe.results)

# %% [markdown]
# ## 3. Run end-to-end
#
# Returns `dict[str, FitResult]`. Each `FitResult` has a Jupyter HTML repr, so simply
# displaying one in a cell gives a small table of ℓ, value, error.

# %%
results = pipe.run()
results["tf_ee_so_90"]

# %% [markdown]
# ## 4. Inline plot + save
#
# Every report plotter returns `(fig, ax)` so notebook use is a one-liner; pass
# `path=` to also write a PNG.

# %%
from cmb_diagnostics.reports import tf as tf_report

fig, ax = tf_report.plot(results, path=None)
fig
