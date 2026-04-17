"""Thin wrapper around scipy.optimize.minimize + Fisher error helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Fitter:
    """Chi^2 minimizer with optional per-point error.

    Unlike V2's ``Estimator.Fitter``, this does NOT run the fit in __init__ --
    call :meth:`fit` explicitly.

    Phase 3/4: port from ``cmb_diagnoistics/Estimator.py::Fitter`` but remove
    the auto-fit side effect from ``__init__``.
    """

    model: Callable[..., np.ndarray] | None = None
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    dy: np.ndarray | None = None

    def fit(self, x0: np.ndarray, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "Phase 4: port from cmb_diagnoistics/Estimator.py::Fitter "
            "(scipy.optimize.minimize on chi^2; no auto-fit in __init__)."
        )


def fisher_error(
    model_derivative: np.ndarray,
    var: np.ndarray,
) -> float:
    """Return the 1-sigma Fisher error on a scalar parameter r given model
    derivative dm/dr and per-point variance.

    Formula: 1 / sqrt(sum(dm_dr**2 / var)).

    Phase 4: port from ``cmb_diagnoistics/diag_utils.py::hess_inv`` (V1 form)
    and ``cmb_diagnoistics/Models.py::rttf_error`` (V2 form, optional via
    cfg.advanced.fisher_error_form).
    """
    raise NotImplementedError(
        "Phase 4: port from cmb_diagnoistics/diag_utils.py::hess_inv "
        "(1/sqrt(sum(dmdr**2/var)), pure scalar Fisher)."
    )
