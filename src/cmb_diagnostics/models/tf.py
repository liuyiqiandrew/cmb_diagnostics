"""Transfer-function amplitude model + rotation model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TFAmplitudeModel:
    """Scalar scaling: predicts ``r * reference(l)``.

    Ports ``cmb_diagnoistics/Models.py::tf_model``.
    """

    def predict(self, r: float, reference: np.ndarray) -> np.ndarray:
        return float(r) * np.asarray(reference)


@dataclass
class RotationModel:
    """(EE - BB) template fit for polarization-angle estimation.

    Ports the template shape used in
    ``cmb_diagnoistics/PolAngEstimator.py::PolAngEB.estimate_pol_ang``
    (``EB = a * (EE - BB)``).
    """

    def predict(self, a: float, ee_minus_bb: np.ndarray) -> np.ndarray:
        return float(a) * np.asarray(ee_minus_bb)


def rotation_from_amplitude(a: float) -> float:
    """Closed-form: ``alpha = arctan(2 a) / 4``.

    Direct port of ``cmb_diagnoistics/diag_utils.py::res2ang``.
    """
    return float(np.arctan(float(a) * 2) / 4)
