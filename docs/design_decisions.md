# Design decisions

`architecture.md` describes *what* the package looks like — the layers, the
dependency direction, the principles. This document explains *why* each major
decision was made, what alternative was rejected, and why the chosen form is
appropriate for this specific package: a narrow, scientifically conservative
calibration pipeline driven from notebooks and the CLI.

The audience is anyone (future-Yiqi, a new collaborator, or a code-review
agent) who is about to change the structure and wants to know which guardrails
were intentional.

## Domain context that shapes the design

A handful of facts about *this* package, not software in general, set the tone
for every decision below:

- The pipeline is small (~30 source files, single-author) and unlikely to
  become large. Over-engineered abstractions cost more than they save here.
- The science kernels are copied verbatim from V1/V2; they are not under
  active redesign. The refactor's job is to expose those kernels behind clean
  seams, not to reinvent them.
- The primary user is a CMB scientist working in Jupyter on Della, iterating
  one mask / one frequency / one configuration at a time. Notebook ergonomics
  matter as much as CLI ergonomics.
- Inputs are large (multi-GB Planck and SO maps) and external dependencies
  (`pymaster`, `pygsm`, `pixell`, `healpy`) are heavy and platform-sensitive.
  Anything that lets a user explore the package without loading the full
  stack is worth doing.
- Reproducibility is not optional. Frozen V2 goldens at `test/bf_tf.npy` /
  `test/ml_tf.npy` are the integration contract; any refactor that breaks
  bit-for-bit agreement is a bug.

Most of the decisions below are direct consequences of one or more of these
facts.

## 1. Typed identifiers (`Tracer`, `SpectrumKey`, `BandInfo`) instead of strings

**Decision.** Frequencies, instruments, spins, and component labels travel
through the package as immutable, hashable dataclasses, not strings.
``Tracer("Planck", 143.0, spin=2)`` replaces V1's ``"p143"`` /
``"s90xp150"`` style keys. ``SpectrumKey(t1, t2, "EE")`` keys the
``Spectra`` container.

**Why for this package.** The V1 code suffered from string-key bugs that
typed checkers cannot catch (``"p143"`` vs ``"P143"`` vs ``"143"``). Because
the same key flows through fields, spectra, models, and reports, a typo at
one level propagated invisibly to the next. Frozen dataclasses make every
identifier mistake a static error or a clean ``KeyError`` at the boundary
where it was introduced — Python finds them in seconds rather than as
mysterious zeros in a fit hours later.

**What we rejected.** Plain ``NamedTuple`` was tempting (faster, lighter)
but loses the ability to add validation or computed properties later. Bare
``str`` is what V1 used and is what we are getting away from. Inheriting
from a base class adds zero value over flat dataclasses for identifiers
that have no behavior of their own.

## 2. Containers separate from logic

**Decision.** ``FieldSet``, ``Spectra``, ``FitResult``, ``CMBReference`` are
small storage classes. They do not compute anything. The work happens in
free functions (``compute_spectra``, ``load_camb_reference``,
``load_mask``) or in stateless estimator classes.

**Why for this package.** V1 had a ``PSContainer`` god-object that owned the
mask, the bins, the fields, the spectra, *and* the fitting. Any change to
one concern forced re-reading the rest. Splitting "what data did I produce"
from "how was it produced" lets each layer be tested independently — the
new ``Spectra`` container has unit tests with synthetic ``add()``-ed
arrays, with no NaMaster anywhere. That same separation makes it possible
to inject hand-built ``Spectra`` into the Pipeline for fast end-to-end
tests (``tests/unit/test_phase4.py::test_pipeline_estimate_tf_ee_with_synthetic_stack``).

**What we rejected.** An ORM-style "fat container" that knows how to load,
save, compute, and plot itself would centralize behavior but couples
storage to every dependency: importing the container would import pymaster
and matplotlib, killing import-time and forcing pymaster onto every
testing environment.

## 3. `Pipeline` exposes step-through methods, not just `run()`

**Decision.** ``Pipeline.run()`` is the one-call entry point, but every
intermediate step is a public method (``load_mask``, ``build_fieldsets``,
``compute_spectra``, ``estimate_tf_ee``, …). Each method populates an
attribute on the Pipeline object so notebook users can pause, inspect,
and re-run.

**Why for this package.** Notebook usage is a first-class concern (see
``docs/architecture.md`` for the "notebook" layer at the top, and the
``notebooks/02_step_through.py`` example). The natural debugging loop for
a CMB calibration analysis is "run up to the spectra, look at them, run
the next step, look at the fit residuals." That requires every
intermediate to be addressable. A single ``run()`` that hides the
intermediates would force users to copy the pipeline body into their
notebook to debug.

**What we rejected.** A purely functional pipeline (``estimate_tf_ee(load_mask(cfg), …)``)
is conceptually clean but burdens the notebook user with bookkeeping every
intermediate by hand. The Pipeline-as-instance pattern lets the user keep
typing ``pipe.``-tab-complete and discover the next step.

## 4. `Config` is a typed dataclass, not a dict

**Decision.** YAML and inline dicts both round-trip through a frozen
``Config`` dataclass tree (``BandpowersConfig``, ``MaskConfig``,
``InstrumentConfig``, …). The CLI calls ``Config.from_yaml``; notebooks
call ``Config.from_dict``.

**Why for this package.** YAML files capture concrete numbers (paths,
frequencies, beams, fit ranges) — exactly the values a reviewer would want
documented next to a result. Validation needs to happen *once*, at the
edge, with a clear error pointing at the offending field. Typed configs
also let the IDE help users (``cfg.bandpowers.bin_width`` autocompletes;
``cfg["bandpwers"]["bin_with"]`` does not). Notebook construction via
``Config.from_dict`` keeps the inline-dict ergonomics without giving up
validation.

**What we rejected.** Pydantic would add an extra dependency and a coercion
layer the package does not need at this size. Loose dicts give up all the
benefits above for one keystroke saved.

## 5. Estimators are stateless after construction

**Decision.** ``TransferFunctionEE(spec_pp=…, spec_ps=…, cmb_ref=…, dust=…)``
captures the inputs at construction. ``estimate(target=…)`` is a pure
function of construction state plus ``target`` — no mutation of ``self``,
no side effects, no plotting. The result is a ``FitResult``.

**Why for this package.** Reproducibility: re-running ``estimate(target=…)``
with the same inputs *must* produce identical numbers. Side effects (a
file write, a counter increment, a cached intermediate) make that
contract brittle. Statelessness also lets a notebook user re-run a single
estimator on a freshly-tweaked dust model without rebuilding everything
above it.

**What we rejected.** Carrying mutable state across calls (``self.last_fit``,
``self.diagnostics_log``) is convenient for one-off scripting but hides
exactly the lifecycle bugs the refactor was meant to surface.

## 6. Protocol-based extensibility, not abstract base classes

**Decision.** ``MapLoader``, ``Estimator``, and ``DustModel`` are
``@runtime_checkable Protocol`` types. There is no ``BaseLoader`` to
inherit from; any class with the right method signature satisfies the
contract.

**Why for this package.** The user-base is small and the extension points
are narrow (one new instrument's loader, one new dust SED). Forcing every
implementation through inheritance would require importing the base class
*and* its module's transitive dependencies. With Protocols, a quick
notebook stub like ``class _NanDust: def predict_cross(...): return 1.0``
satisfies the contract immediately — exactly what
``test_tf_ee_nan_dust_propagates_nan_not_one`` needs.

**What we rejected.** ``ABC`` + ``abstractmethod`` provides "did you implement
the method" checks but loses duck-typed flexibility and adds boilerplate
without real safety improvements over Protocols.

## 7. Frozen dataclasses for identifiers, mutable classes for containers

**Decision.** Anything used as a dict key (``Tracer``, ``SpectrumKey``,
``BandInfo``) is ``frozen=True``. Containers (``FieldSet``, ``Spectra``)
are not frozen — they are populated by ``add()`` calls during pipeline
execution.

**Why for this package.** Frozen identifiers are required for hashability,
which is required for ``dict``-based storage. Containers, by contrast,
are *constructed empty and grown* during pipeline execution; freezing
them would force builders to materialize a complete dict before
construction, doubling memory and complicating partial builds. The
asymmetry is deliberate, not an oversight.

## 8. Lazy imports of heavy / optional dependencies

**Decision.** ``pymaster``, ``pygsm``, ``pixell``, ``healpy``, ``matplotlib``,
and ``camb`` are imported *inside* the function bodies that need them,
not at module top. Type hints that reference these libraries live behind
``if TYPE_CHECKING:``.

**Why for this package.** ``import cmb_diagnostics`` should work in any
environment that has numpy. A notebook user exploring the API on a laptop
without ``pymaster`` should still be able to import ``Tracer``,
``Spectra``, and ``Config`` and read the docstrings. Eager imports would
bind the package to the platform-sensitive scientific stack at the
``import`` statement, breaking docs builds, fast unit tests, and IDE
introspection.

**What we rejected.** A top-level ``try/except ImportError`` per heavy dep,
with stub objects swapped in on failure, is more code than the lazy
pattern and produces opaque errors at use time.

## 9. Single conversion point for unit / convention transforms

**Decision.** Each unit or convention conversion happens *once*, at the
boundary where data enters the pipeline:

- K → μK: in the map loaders (``cfg.unit_scale * 1e6`` at load time).
- CAR → HEALPix: in ``io.loaders.SOCarLoader.load`` via ``pixell.reproject``.
- D_ℓ → C_ℓ: in ``spectra.compute.compute_spectra`` (when ``is_dell=True``)
  and in ``io.camb.load_camb_reference``. Every ``Spectra`` and
  ``CMBReference`` downstream is C_ℓ.

**Why for this package.** Conversion-at-each-consumer is the recipe for the
exact class of bug Finding #2 caught (D_ℓ pseudo-spectra mixed with C_ℓ
models, off by ``ℓ(ℓ+1)/(2π)``). Converting once at the boundary collapses
the surface area of the bug to a single function. Every downstream
consumer can assume canonical units without checking.

**What we rejected.** Storing both representations (``cl_dell`` and
``cl_cell``) is symmetric but doubles memory and adds a fork-in-the-road
at every read. Asking each consumer to convert is what the pre-fix code
did; it does not scale.

## 10. Reports module is downstream of estimators, not embedded

**Decision.** ``reports/tf.py`` and ``reports/pol_angle.py`` consume
``FitResult`` instances. ``FitResult.plot()`` and ``FitResult._repr_html_()``
delegate to the reports module. Estimators do not write files or call
``plt.show``.

**Why for this package.** Plotting is order-of-magnitude slower than
fitting in a notebook session, and matplotlib state leaks between cells.
Keeping plot/IO out of the fit path means a notebook can iterate fits
without producing 30 PNGs, and the integration regression suite can
compare ``FitResult`` arrays bit-for-bit without touching the filesystem.

**What we rejected.** V1 wrote ``debug_*_fit_*.png`` per ℓ-bin into CWD as
a side effect of fitting. The new pipeline preserves the same diagnostic
data on ``FitResult.diagnostics`` and writes plots only when
``cfg.advanced.write_diagnostic_plots`` is set.

## 11. Tests organized by phase, with pure-Python synthetic fixtures

**Decision.** ``tests/unit/test_phase3.py`` covers io/fields/spectra
foundations; ``test_phase4.py`` covers models/estimators; ``test_phase5.py``
covers pipeline/reports. Heavy regression tests against real maps live in
``tests/integration`` behind the ``slow`` marker. Synthetic fixtures
(``bandpowers_factory``, ``spectrumkey_factory``, ``tiny_config``,
``phase3_config`` in ``tests/conftest.py``) let unit tests run end-to-end
without real Planck/SO maps.

**Why for this package.** The full stack needs a real env with pymaster,
pygsm, pixell, and healpy on multi-GB maps. That suite cannot run on a
laptop or in CI. The phase-keyed unit suite runs in ~8 seconds in any env
that has numpy — fast enough to drive TDD on the refactor itself.
``pytest.importorskip("pygsm")`` gates the few tests that genuinely need
the SED library.

**What we rejected.** Mocking pymaster comprehensively (the way Finding #2's
test stubs ``compute_full_master``) is appropriate for a *single* targeted
test but infeasible across the whole suite — pymaster's surface is too
large.

## 12. No backwards-compatibility shims or feature flags

**Decision.** When the refactor changed an interface, we deleted the old
form. There is no ``Spectra.legacy_keys()``, no
``Estimator(use_v1_convention=True)``. The misspelled package
``cmb_diagnoistics/`` was deleted in Phase 6 (it remains on ``main`` for
anyone who needs to reproduce pre-refactor results).

**Why for this package.** The user-base is one person plus future
collaborators, and the V1/V2 implementations live on ``main`` for anyone
who needs them. Maintaining shims doubles the maintenance surface for
zero users. The same logic informed Finding #3 today: the dead positional
``map_{}.fits`` fallback in ``_format_path`` was deleted, not patched.

## 13. CLAUDE.md captures pitfalls that are *not* derivable from code

**Decision.** ``CLAUDE.md`` documents a small set of project-specific
gotchas (effective fsky convention, beam length with pymaster 2.x, the
PYTHONPATH footgun). Architecture, API, and config schemas live in
``docs/`` instead.

**Why for this package.** ``CLAUDE.md`` is loaded into every Claude Code
context for this repo, so it must stay short and high-signal. Anything
that future-you could derive from reading the code (file paths, class
names, function signatures) belongs in ``docs/``, where it can be
generated or updated without bloating the always-loaded preamble.

## When to revisit these decisions

A decision should be re-examined if any of these hold:

- The package crosses ~50 source files. Some of the "small package"
  trade-offs above (no ABCs, no Pydantic, no shims) become more painful at
  scale.
- A second deployment target appears (e.g. running on Princeton's HPC
  *and* on a CI runner without HPC paths). Hard-coded ``yl9946`` paths in
  the example configs should move to env vars or a per-host overlay.
- A second instrument joins SO. The ``InstrumentConfig`` shape and the
  ``MapLoader`` Protocol were designed with one Planck and one SO
  instrument in mind; a third may expose seams worth tightening.
- A non-Knox covariance becomes the default. The ``knox_variance``
  function is positioned to be replaced behind the same call site, but
  the ``Spectra`` container does not yet carry a covariance matrix.
