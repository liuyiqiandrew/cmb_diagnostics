"""Knox-approximation bandpower variance.

Future extension: full NaMaster Gaussian covariance via NmtCovarianceWorkspace
behind the same interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cmb_diagnostics.models.bandpowers import Bandpowers


def knox_variance(
    cl_aa: np.ndarray,
    cl_bb: np.ndarray,
    cl_ab: np.ndarray,
    bandpowers: Bandpowers,
    fsky_effective: float,
    cl_ba: np.ndarray | None = None,
) -> np.ndarray:
    """Bandpower variance under Gaussian, disconnected-trispectrum approximation.

    ``Var(C_AB) = (C_AA * C_BB + C_AB * C_BA) / ((2l+1) * fsky * dl)``.
    For symmetric components (EE, BB, TT, TE), ``cl_ba == cl_ab``; for EB / BE
    pass ``cl_ba`` explicitly.
    """
    cl_ba_arr = cl_ab if cl_ba is None else cl_ba
    ell = bandpowers.effective_ell
    return (cl_aa * cl_bb + cl_ab * cl_ba_arr) / (2 * ell + 1) / fsky_effective / bandpowers.bin_width
