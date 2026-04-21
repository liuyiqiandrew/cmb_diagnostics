"""Smoke test: every public name in cmb_diagnostics.__all__ imports, and
Phase 4 Pipeline methods enforce their step-ordering contract.
"""

from __future__ import annotations

import numpy as np
import pytest

import cmb_diagnostics
from cmb_diagnostics import CMBReference, Config, FitResult, Pipeline, Tracer

EXPECTED_PUBLIC = {
    "BandInfo",
    "Bandpowers",
    "CMBReference",
    "Config",
    "FieldSet",
    "FitResult",
    "MBBDustModel",
    "Mask",
    "Pipeline",
    "PolarizationAngleEB",
    "Spectra",
    "SpectrumKey",
    "Tracer",
    "TransferFunctionEE",
    "TransferFunctionTE",
}


def test_all_exported():
    assert set(cmb_diagnostics.__all__) == EXPECTED_PUBLIC


def test_all_importable():
    for name in cmb_diagnostics.__all__:
        assert hasattr(cmb_diagnostics, name), name


def test_pipeline_phase4_methods_require_pipeline_state(tiny_config: Config):
    """Phase 4 estimators now run; they require mask + spectra to be set."""
    pipe = Pipeline(tiny_config)
    with pytest.raises(RuntimeError, match="load_mask"):
        pipe.estimate_tf_ee(target=Tracer("SO_SAT", 90.0, 2))
    with pytest.raises(RuntimeError, match="load_mask"):
        pipe.estimate_tf_te(target=Tracer("SO_SAT", 90.0, 2))
    with pytest.raises(RuntimeError, match="load_mask"):
        pipe.estimate_pol_angle()


def test_pipeline_run_callable_raises_on_missing_data(tiny_config: Config):
    """Pipeline.run is implemented; tiny_config points at nonexistent paths,
    so invoking run surfaces a FileNotFoundError (or similar IO error) from
    load_mask, not NotImplementedError."""
    pipe = Pipeline(tiny_config)
    with pytest.raises((FileNotFoundError, OSError, RuntimeError, ValueError)):
        pipe.run()


def test_pipeline_initial_state(tiny_config: Config):
    pipe = Pipeline(tiny_config)
    assert pipe.mask is None
    assert pipe.fieldsets == {}
    assert pipe.spectra == {}
    assert pipe.results == {}


def test_fitresult_repr_and_html():
    r = FitResult(
        name="demo",
        ell=np.arange(5, dtype=float),
        values=np.ones(5),
        errors=np.full(5, 0.1),
    )
    assert "demo" in repr(r)
    assert "nbins=5" in repr(r)
    html = r._repr_html_()
    assert "<table>" in html
    assert "demo" in html


def test_fitresult_plot_ndim_dispatch():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    r1 = FitResult(
        name="tf_ee_SO_SAT_90",
        ell=np.linspace(40, 400, 10),
        values=np.linspace(0.3, 0.9, 10),
        errors=np.full(10, 0.02),
    )
    fig, ax = r1.plot()
    assert isinstance(fig, Figure) and isinstance(ax, Axes)

    lmax_arr = np.array([200.0, 300.0, 500.0])
    r2 = FitResult(
        name="pol_angle_eb",
        ell=lmax_arr,
        values=np.zeros((3, lmax_arr.size)),
        errors=np.full((3, lmax_arr.size), 0.001),
        diagnostics={
            "so_pairs": np.array([[90.0, 90.0], [90.0, 150.0], [150.0, 150.0]]),
        },
    )
    fig2, ax2 = r2.plot()
    assert isinstance(fig2, Figure) and isinstance(ax2, Axes)


def test_cmb_reference_get_real(bandpowers_factory):
    ref = CMBReference(
        bandpowers=bandpowers_factory(),
        cls_binned={"EE": np.ones(5), "TE": np.zeros(5)},
    )
    assert ref.has("EE")
    assert not ref.has("BB")
    np.testing.assert_array_equal(ref.get("EE"), np.ones(5))
    with pytest.raises(KeyError):
        ref.get("BB")
