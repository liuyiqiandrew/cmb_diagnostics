"""Transfer-function amplitude model + rotation model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TFAmplitudeModel:
    """Scalar scaling: predicts ``r * reference(l)``.

    Phase 4: fill ``predict`` body. Source: ``cmb_diagnoistics/Models.py::tf_model``.
    """

    def predict(self, r: float, reference: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Phase 4: port from cmb_diagnoistics/Models.py::tf_model "
            "(returns r * reference, trivial)."
        )


@dataclass
class RotationModel:
    """(EE - BB) template fit for polarization-angle estimation.

    Phase 4: fill ``predict``. Source: polarization-angle EB model in
    ``cmb_diagnoistics/PolAngEstimator.py``.
    """

    def predict(self, a: float, ee_minus_bb: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Phase 4: port EB = a * (EE - BB) predictor from "
            "cmb_diagnoistics/PolAngEstimator.py."
        )


def rotation_from_amplitude(a: float) -> float:
    """Closed-form: alpha = arctan(2a) / 4.

    Phase 4: direct port of ``cmb_diagnoistics/diag_utils.py::res2ang``.
    """
    raise NotImplementedError(
        "Phase 4: port from cmb_diagnoistics/diag_utils.py::res2ang "
        "(alpha = arctan(a * 2) / 4)."
    )
