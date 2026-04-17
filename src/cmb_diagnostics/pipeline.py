"""End-to-end Pipeline with step-by-step API for interactive use.

The Pipeline exposes both:
- A monolithic ``run()`` for batch / CLI.
- Per-step methods (``load_mask``, ``build_fieldsets``, ``compute_spectra``,
  ``estimate_tf_ee``, ``estimate_tf_te``, ``estimate_pol_angle``) so notebook
  users can inspect ``self.mask``, ``self.fieldsets``, ``self.spectra``, and
  ``self.results`` at any point.

All step methods raise ``NotImplementedError`` in Phase 2; Phase 4 and Phase 5
fill the bodies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cmb_diagnostics._types import Tracer
from cmb_diagnostics.estimators.base import FitResult

if TYPE_CHECKING:
    from cmb_diagnostics.config import Config
    from cmb_diagnostics.fields.container import FieldSet
    from cmb_diagnostics.io.masks import Mask
    from cmb_diagnostics.models.bandpowers import Bandpowers
    from cmb_diagnostics.models.cmb import CMBReference
    from cmb_diagnostics.spectra.store import Spectra


class Pipeline:
    """Orchestrates a full SO-diagnostics run from a :class:`Config`.

    Intermediate state is stored as attributes so notebook users can inspect
    each step independently:

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
        """Phase 3: load mask + build bandpowers + CAMB reference."""
        raise NotImplementedError(
            "Phase 3: call io.masks.load_mask(cfg.mask) -> self.mask; "
            "Bandpowers.from_config(cfg.bandpowers, cfg.nside) -> self.bandpowers; "
            "io.camb.load_camb_reference(cfg.camb, self.bandpowers) -> self.cmb_ref."
        )

    def build_fieldsets(self) -> dict[str, FieldSet]:
        """Phase 3: populate ``self.fieldsets`` for Planck and SO."""
        raise NotImplementedError(
            "Phase 3: for each of cfg.planck, cfg.so: "
            "fields.build_fieldset(instrument_cfg, self.mask, cfg.nside)."
        )

    def compute_spectra(self) -> dict[str, Spectra]:
        """Phase 3: populate ``self.spectra`` with PP, PS, SS cross containers."""
        raise NotImplementedError(
            "Phase 3: call spectra.compute_spectra(fa, fb, bandpowers, "
            "mask.fsky_effective) for all needed (fa, fb) pairs."
        )

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
