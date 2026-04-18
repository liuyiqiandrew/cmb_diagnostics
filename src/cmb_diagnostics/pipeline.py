"""End-to-end Pipeline with step-by-step API for interactive use.

Phase 3 wires up mask loading, bandpowers, CAMB reference, field construction,
and spectrum computation. Phase 4 will fill the estimator steps; Phase 5 will
compose them into ``run()`` + reports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cmb_diagnostics._types import Tracer
from cmb_diagnostics.estimators.base import FitResult
from cmb_diagnostics.fields.builder import build_fieldset
from cmb_diagnostics.io.camb import load_camb_reference
from cmb_diagnostics.io.masks import load_mask
from cmb_diagnostics.models.bandpowers import Bandpowers
from cmb_diagnostics.spectra.compute import compute_spectra

if TYPE_CHECKING:
    from cmb_diagnostics.config import Config
    from cmb_diagnostics.fields.container import FieldSet
    from cmb_diagnostics.io.masks import Mask
    from cmb_diagnostics.models.cmb import CMBReference
    from cmb_diagnostics.spectra.store import Spectra


class Pipeline:
    """Orchestrates a full SO-diagnostics run from a :class:`Config`.

    Step methods populate attributes so notebook users can inspect each stage:

    .. code-block:: python

        pipe = Pipeline(cfg)
        pipe.load_mask()
        pipe.build_fieldsets()
        pipe.compute_spectra()
        tf90 = pipe.estimate_tf_ee(target=Tracer("SO_SAT", 90.0, spin=2))
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.mask: Mask | None = None
        self.bandpowers: Bandpowers | None = None
        self.cmb_ref: CMBReference | None = None
        self.fieldsets: dict[str, FieldSet] = {}
        self.spectra: dict[str, Spectra] = {}
        self.results: dict[str, FitResult] = {}

    def load_mask(self) -> Mask:
        """Load mask, build bandpowers, parse CAMB reference. Populates
        ``self.mask``, ``self.bandpowers``, ``self.cmb_ref``.
        """
        self.mask = load_mask(self.cfg.mask, self.cfg.nside)
        self.bandpowers = Bandpowers.from_config(self.cfg.bandpowers, self.cfg.nside)
        self.cmb_ref = load_camb_reference(self.cfg.camb, self.bandpowers, self.cfg.nside)
        return self.mask

    def build_fieldsets(self) -> dict[str, FieldSet]:
        """Build Planck and SO ``FieldSet``s. Requires :meth:`load_mask` first."""
        if self.mask is None:
            raise RuntimeError("Pipeline.build_fieldsets: call load_mask() first")
        self.fieldsets["planck"] = build_fieldset(self.cfg.planck, self.mask, self.cfg.nside)
        self.fieldsets["so"] = build_fieldset(self.cfg.so, self.mask, self.cfg.nside)
        return self.fieldsets

    def compute_spectra(self) -> dict[str, Spectra]:
        """Compute PP, PS, SS spectra + Knox variances. Requires fieldsets."""
        if self.mask is None or self.bandpowers is None:
            raise RuntimeError("Pipeline.compute_spectra: call load_mask() first")
        if "planck" not in self.fieldsets or "so" not in self.fieldsets:
            raise RuntimeError("Pipeline.compute_spectra: call build_fieldsets() first")
        fa = self.fieldsets["planck"]
        fb = self.fieldsets["so"]
        fsky = self.mask.fsky_effective
        self.spectra["pp"] = compute_spectra(fa, fa, self.bandpowers, fsky)
        self.spectra["ps"] = compute_spectra(fa, fb, self.bandpowers, fsky)
        self.spectra["ss"] = compute_spectra(fb, fb, self.bandpowers, fsky)
        return self.spectra

    def estimate_tf_ee(self, target: Tracer) -> FitResult:
        """Phase 4: fit EE TF for ``target``."""
        raise NotImplementedError(
            "Phase 4: TransferFunctionEE(self.spectra['pp'], self.spectra['ps'], "
            "self.cmb_ref, dust).estimate(target=target)."
        )

    def estimate_tf_te(self, target: Tracer) -> FitResult:
        """Phase 4: fit TE TF for ``target``."""
        raise NotImplementedError(
            "Phase 4: TransferFunctionTE(...).estimate(target=target)."
        )

    def estimate_pol_angle(self) -> FitResult:
        """Phase 4: fit SO polarization angle from EB."""
        raise NotImplementedError(
            "Phase 4: PolarizationAngleEB(self.spectra['ss'], "
            "lmin=cfg.pol_angle.lmin, lmax_sweep=cfg.pol_angle.lmax_sweep).estimate()."
        )

    def run(self) -> dict[str, FitResult]:
        """Phase 5: run the full pipeline end-to-end.

        Composes the step methods above. Returns ``self.results``.
        """
        raise NotImplementedError(
            "Phase 5: compose load_mask -> build_fieldsets -> compute_spectra -> "
            "estimate_tf_ee (per SO band) -> estimate_tf_te (per SO band) -> "
            "estimate_pol_angle; populate self.results with keyed FitResults."
        )
