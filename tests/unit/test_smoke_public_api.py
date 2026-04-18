"""Smoke test: every public name in cmb_diagnostics.__all__ imports, and Phase 4
behavior still raises NotImplementedError as documented.
"""

from __future__ import annotations

import numpy as np
import pytest

import cmb_diagnostics
from cmb_diagnostics import (
    CMBReference,
    Config,
    FitResult,
    Pipeline,
    PolarizationAngleEB,
    Tracer,
    TransferFunctionEE,
    TransferFunctionTE,
)

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


def test_transfer_function_ee_raises():
    est = TransferFunctionEE(None, None, None, None)
    with pytest.raises(NotImplementedError, match="Phase 4"):
        est.estimate(target=Tracer("SO_SAT", 90.0, 2))


def test_transfer_function_te_raises():
    est = TransferFunctionTE(None, None, None, None, None)
    with pytest.raises(NotImplementedError, match="Phase 4"):
        est.estimate(target=Tracer("SO_SAT", 90.0, 2))


def test_pol_angle_raises():
    est = PolarizationAngleEB(None)
    with pytest.raises(NotImplementedError, match="Phase 4"):
        est.estimate()


def test_pipeline_phase4_methods_raise(tiny_config: Config):
    """Phase 4 estimator methods are still stubs."""
    pipe = Pipeline(tiny_config)
    with pytest.raises(NotImplementedError, match="Phase 4"):
        pipe.estimate_tf_ee(target=Tracer("SO_SAT", 90.0, 2))
    with pytest.raises(NotImplementedError, match="Phase 4"):
        pipe.estimate_tf_te(target=Tracer("SO_SAT", 90.0, 2))
    with pytest.raises(NotImplementedError, match="Phase 4"):
        pipe.estimate_pol_angle()
    with pytest.raises(NotImplementedError, match="Phase 5"):
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
