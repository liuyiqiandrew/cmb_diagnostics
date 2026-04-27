"""End-to-end Pipeline with step-by-step API for interactive use.

Phase 3 wires up mask loading, bandpowers, CAMB reference, field construction,
and spectrum computation. Phase 4 fills the estimator steps; Phase 5 will
compose them into ``run()`` + reports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cmb_diagnostics._types import BandInfo, Tracer
from cmb_diagnostics.estimators.base import FitResult
from cmb_diagnostics.estimators.pol_angle import PolarizationAngleEB
from cmb_diagnostics.estimators.transfer_function import (
    TransferFunctionEE,
    TransferFunctionTE,
)
from cmb_diagnostics.fields.builder import build_fieldset
from cmb_diagnostics.io.camb import load_camb_reference
from cmb_diagnostics.io.masks import load_mask
from cmb_diagnostics.models.bandpowers import Bandpowers
from cmb_diagnostics.models.dust import MBBDustModel
from cmb_diagnostics.spectra.compute import compute_spectra

if TYPE_CHECKING:
    from cmb_diagnostics.config import Config
    from cmb_diagnostics.fields.container import FieldSet
    from cmb_diagnostics.io.masks import Mask
    from cmb_diagnostics.models.cmb import CMBReference
    from cmb_diagnostics.spectra.store import Spectra


class Pipeline:
    """Orchestrate a full SO-diagnostics run from a :class:`Config`.

    Each step populates instance attributes so notebook users can inspect
    every intermediate product:

    .. code-block:: python

        pipe = Pipeline(cfg)
        pipe.load_mask()
        pipe.build_fieldsets()
        pipe.compute_spectra()
        tf90 = pipe.estimate_tf_ee(target=Tracer("SO_SAT", 90.0, spin=2))

    Parameters
    ----------
    cfg : Config
        Typed configuration driving the run.

    Attributes
    ----------
    cfg : Config
        The originating configuration.
    mask : Mask or None
        Populated by :meth:`load_mask`.
    bandpowers : Bandpowers or None
        Populated by :meth:`load_mask`.
    cmb_ref : CMBReference or None
        Populated by :meth:`load_mask`.
    fieldsets : dict of str to FieldSet
        Populated by :meth:`build_fieldsets`; keys ``"planck"`` and ``"so"``.
    spectra : dict of str to Spectra
        Populated by :meth:`compute_spectra`; keys ``"pp"``, ``"ps"``, ``"ss"``.
    results : dict of str to FitResult
        Populated by estimator methods; keyed by ``FitResult.name``.
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
        """Load the analysis mask, build bandpowers, parse CAMB reference.

        Populates ``self.mask``, ``self.bandpowers``, and ``self.cmb_ref``.

        Returns
        -------
        Mask
            The loaded and apodized mask.
        """
        self.mask = load_mask(self.cfg.mask, self.cfg.nside)
        self.bandpowers = Bandpowers.from_config(self.cfg.bandpowers, self.cfg.nside)
        self.cmb_ref = load_camb_reference(self.cfg.camb, self.bandpowers, self.cfg.nside)
        return self.mask

    def build_fieldsets(self) -> dict[str, FieldSet]:
        """Build Planck and SO :class:`FieldSet` objects.

        Requires :meth:`load_mask` to have populated ``self.mask`` first.

        Returns
        -------
        dict of str to FieldSet
            ``{"planck": ..., "so": ...}``.

        Raises
        ------
        RuntimeError
            When :meth:`load_mask` has not yet been called.
        """
        if self.mask is None:
            raise RuntimeError("Pipeline.build_fieldsets: call load_mask() first")
        self.fieldsets["planck"] = build_fieldset(self.cfg.planck, self.mask, self.cfg.nside)
        self.fieldsets["so"] = build_fieldset(self.cfg.so, self.mask, self.cfg.nside)
        return self.fieldsets

    def compute_spectra(self) -> dict[str, Spectra]:
        """Compute Planck x Planck, Planck x SO, and SO x SO spectra.

        All three :class:`Spectra` populate Knox variances at compute time so
        downstream estimators can consume them directly. Requires
        :meth:`build_fieldsets` to have run.

        Returns
        -------
        dict of str to Spectra
            ``{"pp": ..., "ps": ..., "ss": ...}``.

        Raises
        ------
        RuntimeError
            When prerequisites are missing.
        """
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

    def _build_band_info_map(self) -> dict[Tracer, BandInfo]:
        """Build the Tracer->BandInfo map consumed by :class:`MBBDustModel`.

        Spin is set to 2 for both Planck and SO (dust amp fits use EE/BB/TE
        on polarization tracers; T-side tracers reuse the same BandInfo since
        ``eff_freq_dust`` is what ``MBBDustModel`` reads).

        Missing ``eff_freq_cmb`` / ``eff_freq_dust`` fall back to the band's
        nominal ``freq``. SO's wide tophat bands have no calibrated effective
        frequencies; the band center is the honest default.
        """
        out: dict[Tracer, BandInfo] = {}
        for inst in (self.cfg.planck, self.cfg.so):
            for band in inst.bands:
                eff_cmb = band.eff_freq_cmb if band.eff_freq_cmb is not None else band.freq
                eff_dust = band.eff_freq_dust if band.eff_freq_dust is not None else band.freq
                for spin in (0, 2):
                    t = Tracer(inst.name, band.freq, spin=spin)
                    out[t] = BandInfo(
                        tracer=t,
                        beam_fwhm_arcmin=band.beam_fwhm_arcmin,
                        eff_freq_cmb=eff_cmb,
                        eff_freq_dust=eff_dust,
                    )
        return out

    def _ensure_spectra(self) -> None:
        if self.mask is None:
            raise RuntimeError("Pipeline: call load_mask() first")
        missing = [k for k in ("pp", "ps", "ss") if k not in self.spectra]
        if missing:
            raise RuntimeError(
                f"Pipeline: call compute_spectra() first (missing: {missing})"
            )

    def _build_dust(self) -> MBBDustModel:
        return MBBDustModel(
            beta=self.cfg.dust.beta,
            Td_kelvin=self.cfg.dust.Td_kelvin,
            nu0_ghz=self.cfg.dust.nu0_ghz,
            band_info=self._build_band_info_map(),
        )

    def estimate_tf_ee(self, target: Tracer) -> FitResult:
        """Fit the EE transfer function for ``target``.

        Parameters
        ----------
        target : Tracer
            Typically an SO spin-2 tracer.

        Returns
        -------
        FitResult
            The fit result, also stored in ``self.results[result.name]``.
        """
        self._ensure_spectra()
        assert self.cmb_ref is not None  # _ensure_spectra implies load_mask ran
        est = TransferFunctionEE(
            spec_pp=self.spectra["pp"],
            spec_ps=self.spectra["ps"],
            cmb_ref=self.cmb_ref,
            dust=self._build_dust(),
        )
        result = est.estimate(target=target)
        self.results[result.name] = result
        return result

    def estimate_tf_te(self, target: Tracer) -> FitResult:
        """Fit the TE transfer function for ``target``.

        Parameters
        ----------
        target : Tracer
            Typically an SO spin-2 tracer.

        Returns
        -------
        FitResult
            The fit result, also stored in ``self.results[result.name]``.
        """
        self._ensure_spectra()
        assert self.cmb_ref is not None
        est = TransferFunctionTE(
            spec_pp_tt=self.spectra["pp"],
            spec_pp_te=self.spectra["pp"],
            spec_ps_te=self.spectra["ps"],
            cmb_ref=self.cmb_ref,
            dust=self._build_dust(),
        )
        result = est.estimate(target=target)
        self.results[result.name] = result
        return result

    def estimate_pol_angle(self) -> FitResult:
        """Fit the SO polarization angle from ``EB`` over the lmax sweep.

        Returns
        -------
        FitResult
            2-D result (``n_pairs`` x ``n_caps``) with angles in radians,
            also stored in ``self.results[result.name]``.
        """
        self._ensure_spectra()
        est = PolarizationAngleEB(
            spec_ss=self.spectra["ss"],
            lmin=self.cfg.pol_angle.lmin,
            lmax_sweep=self.cfg.pol_angle.lmax_sweep,
        )
        result = est.estimate()
        self.results[result.name] = result
        return result

    def run(self) -> dict[str, FitResult]:
        """Run the full pipeline end-to-end and write all artifacts.

        Composes ``load_mask -> build_fieldsets -> compute_spectra -> per-band
        estimate_tf_ee / estimate_tf_te -> estimate_pol_angle``. Writes
        ``{name}.npz`` for each :class:`FitResult` and three combined ``.png``
        files (``tf_ee.png``, ``tf_te.png``, ``pol_angle.png``) under
        ``cfg.output_dir``.

        When ``cfg.advanced.write_diagnostic_plots`` is truthy, also writes a
        per-result ``{name}_diagnostics.png`` under
        ``cfg.output_dir/diagnostics/`` summarizing the per-ell dust amplitude,
        TF ``r``, and TF chi^2.

        Returns
        -------
        dict of str to FitResult
            ``self.results`` after all estimators have run.
        """
        from pathlib import Path

        import matplotlib

        matplotlib.use("Agg")

        from cmb_diagnostics.reports import pol_angle as _pa_reports
        from cmb_diagnostics.reports import tf as _tf_reports

        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        write_diag = bool(getattr(self.cfg.advanced, "write_diagnostic_plots", False))
        diag_dir = out_dir / "diagnostics" if write_diag else None

        self.load_mask()
        self.build_fieldsets()
        self.compute_spectra()

        tf_ee_results: list[FitResult] = []
        for band in self.cfg.so.bands:
            target = Tracer(self.cfg.so.name, band.freq, spin=2)
            r = self.estimate_tf_ee(target=target)
            _tf_reports.save_npz(r, out_dir / f"{r.name}.npz")
            if diag_dir is not None:
                _tf_reports.plot_diagnostics(r, diag_dir)
            tf_ee_results.append(r)
        if tf_ee_results:
            _tf_reports.plot(tf_ee_results, path=out_dir / "tf_ee.png")

        tf_te_results: list[FitResult] = []
        for band in self.cfg.so.bands:
            target = Tracer(self.cfg.so.name, band.freq, spin=2)
            r = self.estimate_tf_te(target=target)
            _tf_reports.save_npz(r, out_dir / f"{r.name}.npz")
            if diag_dir is not None:
                _tf_reports.plot_diagnostics(r, diag_dir)
            tf_te_results.append(r)
        if tf_te_results:
            _tf_reports.plot(tf_te_results, path=out_dir / "tf_te.png")

        pa = self.estimate_pol_angle()
        _pa_reports.save_npz(pa, out_dir / f"{pa.name}.npz")
        _pa_reports.plot(pa, path=out_dir / "pol_angle.png")

        return self.results
