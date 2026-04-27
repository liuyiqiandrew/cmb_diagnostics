"""Run NaMaster MASTER over all tracer pairs between two FieldSets.

Replaces V1's many ``calc_*`` methods on ``PSContainer`` and V2's
``PSCalculator``. Emits every spectrum component supported by the spin-spin
combination of each pair and attaches per-bin Knox variance.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from cmb_diagnostics._types import SpectrumKey
from cmb_diagnostics.spectra.covariance import knox_variance
from cmb_diagnostics.spectra.store import Spectra

if TYPE_CHECKING:
    from cmb_diagnostics.fields.container import FieldSet
    from cmb_diagnostics.models.bandpowers import Bandpowers


def _pp_auto(field, nmt_bin, dl2cl):
    import pymaster as nmt

    return [c * dl2cl for c in nmt.compute_full_master(field, field, nmt_bin)]  # [EE, EB, BE, BB]


def _tt_auto(field, nmt_bin, dl2cl):
    import pymaster as nmt

    return [c * dl2cl for c in nmt.compute_full_master(field, field, nmt_bin)]  # [TT]


def compute_spectra(
    fa: FieldSet,
    fb: FieldSet,
    bandpowers: Bandpowers,
    fsky_effective: float,
) -> Spectra:
    """Run NaMaster MASTER over every tracer pair between ``fa`` and ``fb``.

    Handled spin combinations
    -------------------------
    - ``(2, 2)``: EE, EB, BE, BB
    - ``(0, 0)``: TT
    - ``(0, 2)``: TE, TB — only emitted for ``(spin-0 in fa, spin-2 in fb)``
      (and the swapped direction when ``fa is not fb``).

    Parameters
    ----------
    fa : FieldSet
        First-side :class:`~cmb_diagnostics.fields.container.FieldSet`.
    fb : FieldSet
        Second-side :class:`~cmb_diagnostics.fields.container.FieldSet`; may
        be ``fa`` (auto) or a different instrument (cross).
    bandpowers : Bandpowers
        Must expose a populated ``nmt_bin``.
    fsky_effective : float
        Apodization-weighted effective sky fraction; fed to
        :func:`knox_variance`.

    Returns
    -------
    Spectra
        All populated spectra and their Knox variances.

    Raises
    ------
    RuntimeError
        When ``bandpowers.nmt_bin`` is ``None``.
    """
    import pymaster as nmt

    if bandpowers.nmt_bin is None:
        raise RuntimeError("compute_spectra requires bandpowers.nmt_bin")
    nmt_bin = bandpowers.nmt_bin
    spec = Spectra(bandpowers=bandpowers)

    # NaMaster returns D_ell when the bin was built with is_Dell=True; convert
    # once at the source so every Spectra and Knox variance downstream is C_ell.
    dl2cl = bandpowers.dl2cl if bandpowers.is_dell else 1.0

    spin2_a = [t for t in fa.tracers(spin=2)]
    spin2_b = [t for t in fb.tracers(spin=2)]
    spin0_a = [t for t in fa.tracers(spin=0)]
    spin0_b = [t for t in fb.tracers(spin=0)]

    pp_auto_a = {t: _pp_auto(fa.get(t), nmt_bin, dl2cl) for t in spin2_a}
    pp_auto_b = pp_auto_a if fa is fb else {t: _pp_auto(fb.get(t), nmt_bin, dl2cl) for t in spin2_b}
    tt_auto_a = {t: _tt_auto(fa.get(t), nmt_bin, dl2cl) for t in spin0_a}
    tt_auto_b = tt_auto_a if fa is fb else {t: _tt_auto(fb.get(t), nmt_bin, dl2cl) for t in spin0_b}

    # spin-2 x spin-2
    for t1, t2 in itertools.product(spin2_a, spin2_b):
        ee, eb, be, bb = (c * dl2cl for c in nmt.compute_full_master(fa.get(t1), fb.get(t2), nmt_bin))
        a_ee = pp_auto_a[t1][0]
        a_bb = pp_auto_a[t1][3]
        b_ee = pp_auto_b[t2][0]
        b_bb = pp_auto_b[t2][3]
        dee = knox_variance(a_ee, b_ee, ee, bandpowers, fsky_effective)
        dbb = knox_variance(a_bb, b_bb, bb, bandpowers, fsky_effective)
        deb = knox_variance(a_ee, b_bb, eb, bandpowers, fsky_effective, cl_ba=be)
        dbe = knox_variance(a_bb, b_ee, be, bandpowers, fsky_effective, cl_ba=eb)
        spec.add(SpectrumKey(t1, t2, "EE"), ee, dee)
        spec.add(SpectrumKey(t1, t2, "EB"), eb, deb)
        spec.add(SpectrumKey(t1, t2, "BE"), be, dbe)
        spec.add(SpectrumKey(t1, t2, "BB"), bb, dbb)

    # spin-0 x spin-0
    for t1, t2 in itertools.product(spin0_a, spin0_b):
        (tt_raw,) = nmt.compute_full_master(fa.get(t1), fb.get(t2), nmt_bin)
        tt = tt_raw * dl2cl
        dtt = knox_variance(tt_auto_a[t1][0], tt_auto_b[t2][0], tt, bandpowers, fsky_effective)
        spec.add(SpectrumKey(t1, t2, "TT"), tt, dtt)

    # spin-0 (fa) x spin-2 (fb): TE, TB
    for t1, t2 in itertools.product(spin0_a, spin2_b):
        te, tb = (c * dl2cl for c in nmt.compute_full_master(fa.get(t1), fb.get(t2), nmt_bin))
        a_tt = tt_auto_a[t1][0]
        b_ee = pp_auto_b[t2][0]
        b_bb = pp_auto_b[t2][3]
        dte = knox_variance(a_tt, b_ee, te, bandpowers, fsky_effective)
        dtb = knox_variance(a_tt, b_bb, tb, bandpowers, fsky_effective)
        spec.add(SpectrumKey(t1, t2, "TE"), te, dte)
        spec.add(SpectrumKey(t1, t2, "TB"), tb, dtb)

    # spin-0 (fb) x spin-2 (fa): TE, TB, with the spin-0 tracer as key1
    # (skip when fa is fb to avoid recomputing the same pairs)
    if fa is not fb:
        for t1, t2 in itertools.product(spin0_b, spin2_a):
            te, tb = (c * dl2cl for c in nmt.compute_full_master(fb.get(t1), fa.get(t2), nmt_bin))
            a_tt = tt_auto_b[t1][0]
            b_ee = pp_auto_a[t2][0]
            b_bb = pp_auto_a[t2][3]
            dte = knox_variance(a_tt, b_ee, te, bandpowers, fsky_effective)
            dtb = knox_variance(a_tt, b_bb, tb, bandpowers, fsky_effective)
            spec.add(SpectrumKey(t1, t2, "TE"), te, dte)
            spec.add(SpectrumKey(t1, t2, "TB"), tb, dtb)

    return spec
