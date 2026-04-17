"""Transfer-function estimators (EE and TE)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cmb_diagnostics._types import Tracer
from cmb_diagnostics.estimators.base import FitResult

if TYPE_CHECKING:
    from cmb_diagnostics.models.cmb import CMBReference
    from cmb_diagnostics.models.dust import DustModel
    from cmb_diagnostics.spectra.store import Spectra


class TransferFunctionEE:
    def __init__(
        self,
        spec_pp: Spectra,
        spec_ps: Spectra,
        cmb_ref: CMBReference,
        dust: DustModel,
    ) -> None:
        self.spec_pp = spec_pp
        self.spec_ps = spec_ps
        self.cmb_ref = cmb_ref
        self.dust = dust

    def estimate(self, target: Tracer) -> FitResult:
        """Fit the EE power-level transfer function TF = r**2 vs target.

        Per l-bin: (1) fit dust amplitude on Planck x Planck residuals; (2) fit
        scalar TF on SO x Planck using CMB + dust as reference. Error via Fisher.

        Phase 4: port from ``cmb_diagnoistics/Estimator.py::SOPlkTF.__tf_ee``
        (with V1 Fisher form, per cfg.advanced.fisher_error_form).
        """
        raise NotImplementedError(
            "Phase 4: port from cmb_diagnoistics/Estimator.py::SOPlkTF.__tf_ee "
            "(per-bin dust fit -> TF fit -> Fisher error -> return FitResult)."
        )

    def _fit_bin(self, i: int, target: Tracer) -> tuple[float, float, dict]:
        """Phase 4: one-bin helper."""
        raise NotImplementedError("Phase 4: per-bin fit helper.")


class TransferFunctionTE:
    def __init__(
        self,
        spec_pp_tt: Spectra,
        spec_pp_te: Spectra,
        spec_ps_te: Spectra,
        cmb_ref: CMBReference,
        dust: DustModel,
    ) -> None:
        self.spec_pp_tt = spec_pp_tt
        self.spec_pp_te = spec_pp_te
        self.spec_ps_te = spec_ps_te
        self.cmb_ref = cmb_ref
        self.dust = dust

    def estimate(self, target: Tracer) -> FitResult:
        """Phase 4: port from V1 ``TransferFuncEstimator.py::TransferFuncTE.calc_tf_te``
        (structurally identical to EE with TE instead of EE and lmin=50).
        """
        raise NotImplementedError(
            "Phase 4: port from cmb_diagnoistics/TransferFuncEstimator.py::"
            "TransferFuncTE.calc_tf_te (V2's __tf_te is a stub; use V1 logic)."
        )
