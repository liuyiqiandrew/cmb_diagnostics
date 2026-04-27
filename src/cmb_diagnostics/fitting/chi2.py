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

    Unlike V2's ``Estimator.Fitter``, this does NOT run the fit in
    ``__init__``; call :meth:`fit` explicitly.

    Parameters
    ----------
    model : callable
        ``model(params, x)`` or, when ``args`` is given,
        ``model(params, x, args)``. Must return a numpy array broadcastable
        with ``y``.
    x : Any
        Independent variable forwarded to ``model``.
    y : numpy.ndarray
        Observed values.
    dy : numpy.ndarray or None, optional
        Per-point 1-sigma errors. When ``None``, unit errors are used
        (equivalent to least-squares).
    args : tuple or None, optional
        Extra positional tuple forwarded to ``model`` after ``x`` — matches
        the V1 ``Models.amp_dust_mbb`` calling convention.
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
        """Minimize the chi^2 starting from ``x0``.

        Parameters
        ----------
        x0 : numpy.ndarray
            Initial parameter guess.
        **kwargs
            Forwarded to :func:`scipy.optimize.minimize`.

        Returns
        -------
        scipy.optimize.OptimizeResult
            Result of the minimization; ``res.x`` holds the best-fit params.
        """
        return opt.minimize(self._chi2, np.atleast_1d(x0), **kwargs)


def fisher_error(model_derivative: np.ndarray, var: np.ndarray) -> float:
    """1-sigma Fisher error on a scalar parameter.

    Computes ``1 / sqrt(sum(dm_dr ** 2 / var))``. Equivalent to
    ``sqrt(diag_utils.hess_inv(dm_dr, var))`` and
    ``Models.rttf_error(dm_dr, sqrt(var))`` from V1.

    Parameters
    ----------
    model_derivative : numpy.ndarray
        Derivative of the model with respect to the parameter, evaluated at
        each data point.
    var : numpy.ndarray
        Per-point variance (not sigma).

    Returns
    -------
    float
        1-sigma Fisher error on the parameter.
    """
    fisher = (np.asarray(model_derivative) ** 2 / np.asarray(var)).sum()
    return float(1.0 / np.sqrt(fisher))
