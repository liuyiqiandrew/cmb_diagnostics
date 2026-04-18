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

    def _eff_dust_freq(self, t: Tracer, override: BandInfoMap | None) -> float:
        source = override if override else self.band_info
        info = source.get(t)
        if info is None or info.eff_freq_dust is None:
            raise KeyError(
                f"MBBDustModel needs band_info[{t}].eff_freq_dust; provide one via "
                "MBBDustModel(band_info=...) or pass band_info= to predict_cross."
            )
        return float(info.eff_freq_dust)

    def predict_cross(
        self, t1: Tracer, t2: Tracer, band_info: BandInfoMap | None = None
    ) -> float:
        """Scalar MBB factor for the (t1, t2) pair at unit amplitude.

        Matches V1 ``Models.amp_dust_mbb`` with ``amp=1``: the product of the
        power-law and modified-blackbody factors referenced to ``nu0``, scaled
        by ``trj2tcmb(f1) * trj2tcmb(f2)``.
        """
        from pygsm import planck_law, trj2tcmb

        f1 = self._eff_dust_freq(t1, band_info)
        f2 = self._eff_dust_freq(t2, band_info)
        nu0 = self.nu0_ghz
        beta = self.beta
        Td = self.Td_kelvin

        r2c_f1 = trj2tcmb(f1)
        r2c_f2 = trj2tcmb(f2)
        mbb_pl = (f1 * f2 / nu0 ** 2) ** beta
        mbb_bb = planck_law(Td, f1) * planck_law(Td, f2) / planck_law(Td, nu0) ** 2
        return float(mbb_pl * mbb_bb * r2c_f1 * r2c_f2)

    def fit_amplitude(
        self,
        spectra: Spectra,
        cmb_ref: CMBReference,
        tracer_pairs: Iterable[tuple[Tracer, Tracer]],
        comp: Comp,
        ell_idx: int,
    ) -> FitAmplitude:
        """Phase 4: per-bin dust-amplitude fit (see ``SOPlkTF.__tf_ee``)."""
        raise NotImplementedError(
            "Phase 4: port dust-amplitude fit stage from "
            "cmb_diagnoistics/Estimator.py::SOPlkTF.__tf_ee."
        )
