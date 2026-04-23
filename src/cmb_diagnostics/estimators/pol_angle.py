"""Polarization-angle estimator from SO EB."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from cmb_diagnostics.estimators.base import FitResult
from cmb_diagnostics.fitting.chi2 import Fitter, fisher_error
from cmb_diagnostics.models.tf import RotationModel, rotation_from_amplitude

if TYPE_CHECKING:
    from cmb_diagnostics._types import Tracer
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
        bp = self.spec_ss.bandpowers
        if bp is None:
            raise RuntimeError("PolarizationAngleEB requires spec_ss.bandpowers")
        eff_ell = bp.effective_ell

        so_tracers = sorted(
            {t for pair in self.spec_ss.iter_pairs("EE") for t in pair if t.spin == 2},
            key=lambda t: (t.instrument, t.freq),
        )
        so_pairs: list[tuple[Tracer, Tracer]] = list(
            itertools.combinations_with_replacement(so_tracers, 2)
        )
        n_pairs = len(so_pairs)
        n_caps = len(self.lmax_sweep)

        ang = np.zeros((n_pairs, n_caps))
        var = np.zeros((n_pairs, n_caps))

        rot = RotationModel()

        def model(params: np.ndarray, x: np.ndarray) -> np.ndarray:
            return rot.predict(float(params[0]), x)

        for j, lmax in enumerate(self.lmax_sweep):
            cap_msk = (eff_ell < lmax) & (eff_ell > self.lmin)
            for i, (t1, t2) in enumerate(so_pairs):
                ee, _ = self.spec_ss.get(t1, t2, "EE")
                bb, _ = self.spec_ss.get(t1, t2, "BB")
                eb, eb_var = self.spec_ss.get(t1, t2, "EB")
                if eb_var is None:
                    raise ValueError(
                        f"PolarizationAngleEB: no variance for ({t1}, {t2}, 'EB'); "
                        "compute_spectra must populate Knox variances before fitting."
                    )

                # Empty / near-empty window: one free parameter needs ≥2 bins for
                # a meaningful Fisher error; otherwise the optimizer just returns
                # its starting guess and fisher_error diverges.
                if cap_msk.sum() < 2:
                    ang[i, j] = float("nan")
                    var[i, j] = float("nan")
                    continue

                template = (ee - bb)[cap_msk]
                eb_c = eb[cap_msk]
                eb_var_c = eb_var[cap_msk]

                fitter = Fitter(model=model, x=template, y=eb_c, dy=np.sqrt(eb_var_c))
                res = fitter.fit(x0=np.array([1.0]))
                a = float(res.x[0])

                alpha = rotation_from_amplitude(a)
                a_fisher = fisher_error(template, eb_var_c)
                # alpha = arctan(2a)/4 → dalpha/da = 1 / (2 (1 + 4 a^2)).
                # Finite at a=0 (unlike V1's alpha/a form).
                dalpha_da = 1.0 / (2.0 * (1.0 + 4.0 * a * a))
                alpha_var = (a_fisher * dalpha_da) ** 2

                ang[i, j] = alpha
                var[i, j] = alpha_var

        pair_labels = np.array(
            [[t1.freq, t2.freq] for (t1, t2) in so_pairs], dtype=float
        )
        lmax_arr = np.asarray(self.lmax_sweep, dtype=float)
        return FitResult(
            name="pol_angle_eb",
            ell=lmax_arr,
            values=ang,
            errors=np.sqrt(var),
            diagnostics={"so_pairs": pair_labels, "lmax_sweep": lmax_arr, "var": var},
            metadata={"lmin": self.lmin, "estimator": "PolarizationAngleEB"},
        )
