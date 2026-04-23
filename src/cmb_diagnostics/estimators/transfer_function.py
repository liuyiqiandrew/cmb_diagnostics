"""Transfer-function estimators (EE and TE)."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

from cmb_diagnostics._types import Tracer
from cmb_diagnostics.estimators.base import FitResult
from cmb_diagnostics.fitting.chi2 import Fitter, fisher_error
from cmb_diagnostics.models.tf import TFAmplitudeModel

if TYPE_CHECKING:
    from cmb_diagnostics.models.cmb import CMBReference
    from cmb_diagnostics.models.dust import DustModel
    from cmb_diagnostics.spectra.store import Spectra


def _planck_tracers(spectra: Spectra, comp: str, spin: int) -> list[Tracer]:
    """Unique tracers at ``spin`` that participate in a ``comp`` spectrum."""
    seen: set[Tracer] = set()
    for t1, t2 in spectra.iter_pairs(comp):
        if t1.spin == spin:
            seen.add(t1)
        if t2.spin == spin:
            seen.add(t2)
    return sorted(seen, key=lambda t: (t.instrument, t.freq))


def _fit_tf_bin(
    dust: DustModel,
    spec_pp: Spectra,
    spec_ps: Spectra,
    cmb_ref: CMBReference,
    pp_pairs: list[tuple[Tracer, Tracer]],
    ps_pairs: list[tuple[Tracer, Tracer]],
    comp: str,
    ell_idx: int,
) -> tuple[float, float, float, float]:
    """Fit dust amplitude then TF amplitude at a single l-bin.

    Returns ``(r, dtf, a_dust, chi2_tf)``.
    """
    # EE drops negative-residual pairs (V2 convention); TE/BB keep them (V1 TE).
    dfit = dust.fit_amplitude(
        spec_pp, cmb_ref, pp_pairs, comp=comp, ell_idx=ell_idx,
        positive_only=(comp == "EE"),
    )

    # Short-circuit when the dust step is undefined (no surviving Planck pairs):
    # downstream `ref_plk` would be all-NaN, Fitter.fit would return the initial
    # guess r=1.0, and tf[i] would silently read back as 1.0.
    if not np.isfinite(dfit.value):
        nan = float("nan")
        return nan, nan, float(dfit.value), nan

    cmb_at_ell = float(cmb_ref.get(comp)[ell_idx])
    dust_unity_ps = np.array([dust.predict_cross(t1, t2) for (t1, t2) in ps_pairs])
    ref_plk = dfit.value * dust_unity_ps + cmb_at_ell

    pxs = np.zeros(len(ps_pairs))
    var = np.zeros(len(ps_pairs))
    for k, (t1, t2) in enumerate(ps_pairs):
        cl_k, var_k = spec_ps.get(t1, t2, comp)
        if var_k is None:
            raise ValueError(
                f"TF estimator: no variance for ({t1}, {t2}, {comp!r}); "
                "compute_spectra must populate Knox variances before fitting."
            )
        pxs[k] = cl_k[ell_idx]
        var[k] = var_k[ell_idx]

    tf_m = TFAmplitudeModel()

    def model(params: np.ndarray, x: np.ndarray) -> np.ndarray:
        return tf_m.predict(float(params[0]), x)

    fitter = Fitter(model=model, x=ref_plk, y=pxs, dy=np.sqrt(var))
    res = fitter.fit(x0=np.array([1.0]))
    r = float(res.x[0])

    err_r = fisher_error(ref_plk, var)
    dtf = 2 * abs(r) * err_r
    return r, dtf, float(dfit.value), float(res.fun)


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
        bp = self.spec_pp.bandpowers
        if bp is None:
            raise RuntimeError("TransferFunctionEE requires spec_pp.bandpowers")
        msk = bp.msk
        eff_ell = bp.effective_ell
        bin_indices = np.where(msk)[0]
        n_bins = int(bin_indices.size)

        plk_tracers = _planck_tracers(self.spec_pp, "EE", spin=2)
        pp_pairs = list(itertools.combinations(plk_tracers, 2))
        ps_pairs = [(pt, target) for pt in plk_tracers]

        tf = np.zeros(n_bins)
        dtf = np.zeros(n_bins)
        r_hat = np.zeros(n_bins)
        a_hat = np.zeros(n_bins)
        chi2 = np.zeros(n_bins)

        for i, ell_idx in enumerate(bin_indices):
            r, d, a, c = _fit_tf_bin(
                self.dust, self.spec_pp, self.spec_ps, self.cmb_ref,
                pp_pairs, ps_pairs, comp="EE", ell_idx=int(ell_idx),
            )
            tf[i] = r ** 2
            dtf[i] = d
            r_hat[i] = r
            a_hat[i] = a
            chi2[i] = c

        return FitResult(
            name=f"tf_ee_{target.instrument}_{int(target.freq)}",
            ell=eff_ell[msk],
            values=tf,
            errors=dtf,
            diagnostics={"r": r_hat, "dust_amp": a_hat, "chi2_tf": chi2},
            metadata={"target": target, "estimator": "TransferFunctionEE"},
        )


class TransferFunctionTE:
    def __init__(
        self,
        spec_pp_tt: Spectra,
        spec_pp_te: Spectra,
        spec_ps_te: Spectra,
        cmb_ref: CMBReference,
        dust: DustModel,
        lmin: int | None = 50,
    ) -> None:
        # spec_pp_tt retained in the signature for forward compatibility but
        # is not consumed: Knox variances on TE are already populated on
        # spec_pp_te at compute-time.
        self.spec_pp_tt = spec_pp_tt
        self.spec_pp_te = spec_pp_te
        self.spec_ps_te = spec_ps_te
        self.cmb_ref = cmb_ref
        self.dust = dust
        self.lmin = lmin

    def estimate(self, target: Tracer) -> FitResult:
        bp = self.spec_pp_te.bandpowers
        if bp is None:
            raise RuntimeError("TransferFunctionTE requires spec_pp_te.bandpowers")
        eff_ell = bp.effective_ell
        msk = bp.msk.copy()
        if self.lmin is not None:
            msk &= eff_ell > self.lmin
        bin_indices = np.where(msk)[0]
        n_bins = int(bin_indices.size)

        planck_t = _planck_tracers(self.spec_pp_te, "TE", spin=0)
        planck_e = _planck_tracers(self.spec_pp_te, "TE", spin=2)

        # 12 ordered (T, E) Planck pairs with different frequency, per V1.
        pp_pairs = [
            (t0, t2) for t0, t2 in itertools.product(planck_t, planck_e)
            if not np.isclose(t0.freq, t2.freq)
        ]
        ps_pairs = [(t0, target) for t0 in planck_t]

        tf = np.zeros(n_bins)
        dtf = np.zeros(n_bins)
        r_hat = np.zeros(n_bins)
        a_hat = np.zeros(n_bins)
        chi2 = np.zeros(n_bins)

        for i, ell_idx in enumerate(bin_indices):
            r, d, a, c = _fit_tf_bin(
                self.dust, self.spec_pp_te, self.spec_ps_te, self.cmb_ref,
                pp_pairs, ps_pairs, comp="TE", ell_idx=int(ell_idx),
            )
            tf[i] = r ** 2
            dtf[i] = d
            r_hat[i] = r
            a_hat[i] = a
            chi2[i] = c

        return FitResult(
            name=f"tf_te_{target.instrument}_{int(target.freq)}",
            ell=eff_ell[msk],
            values=tf,
            errors=dtf,
            diagnostics={"r": r_hat, "dust_amp": a_hat, "chi2_tf": chi2},
            metadata={"target": target, "estimator": "TransferFunctionTE"},
        )
