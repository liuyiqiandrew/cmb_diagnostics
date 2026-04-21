"""Thin wrapper around scipy.optimize.minimize + Fisher error helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.optimize as opt


@dataclass
class Fitter:
    """Chi^2 minimizer with optional per-point error.

    Unlike V2's ``Estimator.Fitter``, this does NOT run the fit in __init__ --
    call :meth:`fit` explicitly. ``args`` is an optional extra positional tuple
    forwarded to ``model(params, x, args)`` (matches V1 ``Models.amp_dust_mbb``).
    """

    model: Callable[..., np.ndarray]
    x: Any
    y: np.ndarray
    dy: np.ndarray | None = None
    args: tuple | None = None

    def _chi2(self, params: np.ndarray) -> float:
        if self.args is not None:
            pred = self.model(params, self.x, self.args)
        else:
            pred = self.model(params, self.x)
        dy = self.dy if self.dy is not None else np.ones_like(self.y)
        return float(((self.y - pred) ** 2 / dy ** 2).sum())

    def fit(self, x0: np.ndarray, **kwargs: Any) -> opt.OptimizeResult:
        return opt.minimize(self._chi2, np.atleast_1d(x0), **kwargs)


def fisher_error(model_derivative: np.ndarray, var: np.ndarray) -> float:
    """1-sigma Fisher error on a scalar parameter r given dm/dr and per-point
    variance.

    Formula: ``1 / sqrt(sum(dm_dr**2 / var))``. Equivalent to
    ``sqrt(diag_utils.hess_inv(dm_dr, var))`` / ``Models.rttf_error(dm_dr, sqrt(var))``.
    """
    fisher = (np.asarray(model_derivative) ** 2 / np.asarray(var)).sum()
    return float(1.0 / np.sqrt(fisher))
