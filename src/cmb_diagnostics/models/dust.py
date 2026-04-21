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
        positive_only: bool | None = None,
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
        f1 = self._eff_dust_freq(t1, band_info)
        f2 = self._eff_dust_freq(t2, band_info)
        nu0 = self.nu0_ghz
        beta = self.beta
        Td = self.Td_kelvin

        from pygsm import planck_law, trj2tcmb

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
        positive_only: bool | None = None,
    ) -> FitAmplitude:
        """Per-bin chi^2 fit of the dust amplitude ``a`` at bin ``ell_idx``.

        Model at this bin:
          ``C_ell(t1, t2) = a * predict_cross(t1, t2) + cmb_ref[comp][ell_idx]``

        ``positive_only`` drops pairs with non-positive residual
        (``C - CMB <= 0``). When ``None``, defaults to ``True`` for EE
        (matches V2 ``Estimator.py:130``) and ``False`` otherwise — V1
        ``TransferFuncTE`` does not filter negative TE residuals. Pairs
        with non-positive variance are always dropped.
        """
        from cmb_diagnostics.fitting.chi2 import Fitter, fisher_error

        pairs = list(tracer_pairs)
        dust_unity = np.array([self.predict_cross(t1, t2) for (t1, t2) in pairs])
        cmb_at_ell = float(cmb_ref.get(comp)[ell_idx])

        y = np.zeros(len(pairs))
        var = np.zeros(len(pairs))
        for k, (t1, t2) in enumerate(pairs):
            cl, v = spectra.get(t1, t2, comp)
            if v is None:
                raise ValueError(
                    f"MBBDustModel.fit_amplitude: no variance for ({t1}, {t2}, {comp!r}); "
                    "Knox variances must be populated before fitting."
                )
            y[k] = cl[ell_idx] - cmb_at_ell
            var[k] = v[ell_idx]

        if positive_only is None:
            positive_only = comp == "EE"
        keep = var > 0
        if positive_only:
            keep &= y > 0
        if keep.sum() < 1:
            return FitAmplitude(value=float("nan"), error=float("nan"), chi2=float("nan"))

        y_fit = y[keep]
        du_fit = dust_unity[keep]
        var_fit = var[keep]

        def model(params: np.ndarray, x: np.ndarray) -> np.ndarray:
            return params[0] * x

        fitter = Fitter(model=model, x=du_fit, y=y_fit, dy=np.sqrt(var_fit))
        res = fitter.fit(x0=np.array([1.0]))
        a_hat = float(res.x[0])
        a_err = fisher_error(du_fit, var_fit)
        chi2_min = float(res.fun)
        return FitAmplitude(value=a_hat, error=a_err, chi2=chi2_min)
