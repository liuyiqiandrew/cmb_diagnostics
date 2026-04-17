"""Modified black-body dust amplitude model."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from cmb_diagnostics._types import BandInfo, Comp, Tracer

if TYPE_CHECKING:
    from cmb_diagnostics.models.cmb import CMBReference
    from cmb_diagnostics.spectra.store import Spectra


BandInfoMap = dict[Tracer, BandInfo]


@dataclass
class FitAmplitude:
    value: float
    error: float
    chi2: float


@runtime_checkable
class DustModel(Protocol):
    def predict_cross(
        self, t1: Tracer, t2: Tracer, band_info: BandInfoMap
    ) -> np.ndarray: ...

    def fit_amplitude(
        self,
        spectra: Spectra,
        cmb_ref: CMBReference,
        tracer_pairs: Iterable[tuple[Tracer, Tracer]],
        comp: Comp,
        ell_idx: int,
    ) -> FitAmplitude: ...


@dataclass
class MBBDustModel:
    beta: float = 1.6
    Td_kelvin: float = 19.6
    nu0_ghz: float = 353.0
    band_info: BandInfoMap = field(default_factory=dict)

    def predict_cross(
        self, t1: Tracer, t2: Tracer, band_info: BandInfoMap | None = None
    ) -> np.ndarray:
        """Predict the MBB dust Cl shape for (t1, t2) at unit amplitude.

        Phase 3: port from ``cmb_diagnoistics/Models.py::amp_dust_mbb`` and
        ``cmb_diagnoistics/diag_utils.py::dust_dl``.
        """
        raise NotImplementedError(
            "Phase 3: port from cmb_diagnoistics/Models.py::amp_dust_mbb "
            "(MBB shape at unit amplitude, using pygsm.trj2tcmb and planck_law)."
        )

    def fit_amplitude(
        self,
        spectra: Spectra,
        cmb_ref: CMBReference,
        tracer_pairs: Iterable[tuple[Tracer, Tracer]],
        comp: Comp,
        ell_idx: int,
    ) -> FitAmplitude:
        """Fit the dust amplitude at one bandpower across given tracer pairs.

        Phase 4: port from inner loop of ``cmb_diagnoistics/Estimator.py::SOPlkTF.__tf_ee``
        (dust amplitude stage).
        """
        raise NotImplementedError(
            "Phase 4: port dust-amplitude fit stage from "
            "cmb_diagnoistics/Estimator.py::SOPlkTF.__tf_ee."
        )
