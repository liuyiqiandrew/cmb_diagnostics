"""Run NaMaster MASTER over all tracer pairs between two FieldSets."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cmb_diagnostics.fields.container import FieldSet
    from cmb_diagnostics.models.bandpowers import Bandpowers
    from cmb_diagnostics.spectra.store import Spectra


def compute_spectra(
    fa: FieldSet,
    fb: FieldSet,
    bandpowers: Bandpowers,
    fsky_effective: float,
) -> Spectra:
    """Compute all cross-spectra between ``fa`` and ``fb`` at ``bandpowers``.

    Phase 3: port from V2 ``PSCalculator.PSCalculator`` (which runs
    ``nmt.compute_full_master`` for each tracer pair and attaches Knox
    variance). Autos are computed on a per-tracer basis as needed for the
    Knox variance of cross pairs.

    Returns a :class:`Spectra` keyed by :class:`SpectrumKey` with EE/EB/BE/BB
    for spin-2 x spin-2 pairs, TE/TB for spin-0 x spin-2, and TT for
    spin-0 x spin-0. Also attaches per-bin Knox variance via
    :func:`covariance.knox_variance`.
    """
    raise NotImplementedError(
        "Phase 3: port from cmb_diagnoistics/PSCalculator.py::PSCalculator "
        "(nmt.compute_full_master per tracer pair + Knox variance attachment)."
    )
