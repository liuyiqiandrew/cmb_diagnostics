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
    """Result of a single dust-amplitude fit at one bin.

    Parameters
    ----------
    value : float
        Best-fit dust amplitude (NaN when the fit is undefined).
    error : float
        1-sigma Fisher error on ``value`` (NaN when undefined).
    chi2 : float
        Minimum chi^2 achieved by the fit.
    """

    value: float
    error: float
    chi2: float


@runtime_checkable
class DustModel(Protocol):
    """Structural protocol for dust models consumed by the TF estimators.

    Any class exposing the two methods below (with compatible signatures)
    satisfies this protocol at runtime via ``isinstance(..., DustModel)``.
    """

    def predict_cross(
        self, t1: Tracer, t2: Tracer, band_info: BandInfoMap
    ) -> np.ndarray:
        """Return the unit-amplitude SED factor for a tracer pair."""
        ...

    def fit_amplitude(
        self,
        spectra: Spectra,
        cmb_ref: CMBReference,
        tracer_pairs: Iterable[tuple[Tracer, Tracer]],
        comp: Comp,
        ell_idx: int,
        positive_only: bool | None = None,
    ) -> FitAmplitude:
        """Fit the dust amplitude at one bin and return the :class:`FitAmplitude`."""
        ...


@dataclass
class MBBDustModel:
    """Modified-blackbody dust model with per-tracer effective frequencies.

    Parameters
    ----------
    beta : float, optional
        Dust spectral index. Default ``1.6``.
    Td_kelvin : float, optional
        Dust temperature in Kelvin. Default ``19.6``.
    nu0_ghz : float, optional
        Reference frequency in GHz. Default ``353.0``.
    band_info : dict of Tracer to BandInfo, optional
        Per-tracer effective frequencies; entries must carry a non-None
        ``eff_freq_dust`` or :meth:`predict_cross` raises ``KeyError``.
    """

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
        """Unit-amplitude MBB factor for the ``(t1, t2)`` cross-spectrum.

        Matches V1 ``Models.amp_dust_mbb`` with ``amp = 1``: the product of the
        power-law and modified-blackbody factors referenced to ``nu0``, scaled
        by ``trj2tcmb(f1) * trj2tcmb(f2)``.

        Parameters
        ----------
        t1 : Tracer
            First tracer; ``eff_freq_dust`` looked up from ``band_info``.
        t2 : Tracer
            Second tracer.
        band_info : dict of Tracer to BandInfo or None, optional
            Override for ``self.band_info``; when ``None``, ``self.band_info``
            is used.

        Returns
        -------
        float
            Scalar SED factor for this pair.

        Raises
        ------
        KeyError
            When either tracer has no ``eff_freq_dust``.
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
        """Fit the dust amplitude ``a`` at a single bin by chi^2 minimization.

        Model at this bin::

            C_ell(t1, t2) = a * predict_cross(t1, t2) + cmb_ref[comp][ell_idx]

        Parameters
        ----------
        spectra : Spectra
            Container with per-pair ``C_ell`` and Knox variances.
        cmb_ref : CMBReference
            Binned CMB reference used as the baseline.
        tracer_pairs : iterable of tuple of Tracer
            Pairs to include in the fit.
        comp : Comp
            Spectrum component (``"EE"``, ``"TE"``, ...).
        ell_idx : int
            Index into the binned ell axis.
        positive_only : bool or None, optional
            If ``True``, drop pairs with non-positive residual
            (``C - CMB <= 0``). When ``None``, defaults to ``True`` for
            ``"EE"`` (matches V2 ``Estimator.py:130``) and ``False`` otherwise.
            Pairs with non-positive variance are always dropped.

        Returns
        -------
        FitAmplitude
            Best-fit amplitude, Fisher error, and minimum chi^2. All fields
            are NaN when fewer than one pair survives the variance/positivity
            filter.

        Raises
        ------
        ValueError
            When a selected pair has no populated variance entry.
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
            """Linear amplitude model ``a * x`` for the single-parameter fit."""
            return params[0] * x

        fitter = Fitter(model=model, x=du_fit, y=y_fit, dy=np.sqrt(var_fit))
        res = fitter.fit(x0=np.array([1.0]))
        a_hat = float(res.x[0])
        a_err = fisher_error(du_fit, var_fit)
        chi2_min = float(res.fun)
        return FitAmplitude(value=a_hat, error=a_err, chi2=chi2_min)
