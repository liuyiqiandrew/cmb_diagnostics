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
) -> np.ndarray:
    """Bandpower variance under Gaussian, disconnected-trispectrum approximation.

    Phase 3: port from ``cmb_diagnoistics/diag_utils.py::knox_covar``.

    Formula: ``Var(C_AB) = (C_AA * C_BB + C_AB**2) / ((2l+1) * fsky * dl)``
    with ``l`` = effective bin center from ``bandpowers``.
    """
    raise NotImplementedError(
        "Phase 3: port from cmb_diagnoistics/diag_utils.py::knox_covar "
        "(formula given in docstring; uses bandpowers.effective_ell and bin_width)."
    )
