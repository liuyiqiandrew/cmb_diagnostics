"""Polarization-angle estimator from SO EB."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from cmb_diagnostics.estimators.base import FitResult

if TYPE_CHECKING:
    from cmb_diagnostics.spectra.store import Spectra


class PolarizationAngleEB:
    def __init__(
        self,
        spec_ss: Spectra,
        lmin: int = 30,
        lmax_sweep: Sequence[int] = (200, 250, 300, 350, 400, 450, 500),
    ) -> None:
        self.spec_ss = spec_ss
        self.lmin = lmin
        self.lmax_sweep = tuple(lmax_sweep)

    def estimate(self) -> FitResult:
        """Fit EB = a * (EE - BB); alpha = arctan(2a)/4. Sweep lmax_sweep upper caps.

        Phase 4: port from ``cmb_diagnoistics/PolAngEstimator.py::PolAngEB.calc_pol_ang``.
        """
        raise NotImplementedError(
            "Phase 4: port from cmb_diagnoistics/PolAngEstimator.py::"
            "PolAngEB.calc_pol_ang (sweep lmax caps, fit EB template, return FitResult)."
        )
