"""Transfer-function amplitude model + rotation model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TFAmplitudeModel:
    """Scalar transfer-function amplitude: ``C_ell = r * reference(ell)``."""

    def predict(self, r: float, reference: np.ndarray) -> np.ndarray:
        """Scale a reference spectrum by the TF amplitude.

        Parameters
        ----------
        r : float
            Scalar TF amplitude.
        reference : numpy.ndarray
            Reference spectrum (typically ``a_dust * dust_template + CMB``).

        Returns
        -------
        numpy.ndarray
            ``r * reference``.
        """
        return float(r) * np.asarray(reference)


@dataclass
class RotationModel:
    """(EE - BB) template for pol-angle: ``EB = a * (EE - BB)``."""

    def predict(self, a: float, ee_minus_bb: np.ndarray) -> np.ndarray:
        """Evaluate the rotation template at amplitude ``a``.

        Parameters
        ----------
        a : float
            Rotation amplitude.
        ee_minus_bb : numpy.ndarray
            Precomputed ``EE - BB`` template.

        Returns
        -------
        numpy.ndarray
            ``a * (EE - BB)``.
        """
        return float(a) * np.asarray(ee_minus_bb)


def rotation_from_amplitude(a: float) -> float:
    """Convert a rotation amplitude to a polarization angle.

    Closed form ``alpha = arctan(2 * a) / 4`` — finite at ``a = 0`` unlike
    the V1 ``alpha / a`` form.

    Parameters
    ----------
    a : float
        Fit rotation amplitude from ``EB = a * (EE - BB)``.

    Returns
    -------
    float
        Polarization angle ``alpha`` in radians.
    """
    return float(np.arctan(float(a) * 2) / 4)
