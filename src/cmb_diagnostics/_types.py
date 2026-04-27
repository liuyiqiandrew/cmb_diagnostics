"""Identifier dataclasses shared across the package.

Kept in one module so every submodule has a single import source and so circular
imports between ``fields``, ``spectra``, ``models``, and ``estimators`` cannot
arise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Comp = Literal["EE", "EB", "BE", "BB", "TE", "TB", "TT"]


@dataclass(frozen=True)
class Tracer:
    """Immutable identifier for a single instrument/frequency/spin view.

    Replaces V1/V2's ad-hoc string keys like ``"s90xp150"`` with a typed,
    hashable tuple so Python catches typos statically.

    Parameters
    ----------
    instrument : str
        Instrument name, e.g. ``"Planck"`` or ``"SO_SAT"``.
    freq : float
        Band center frequency in GHz (not necessarily an integer — pass the
        nominal band value, e.g. ``143.0``).
    spin : int
        NaMaster field spin: ``0`` for temperature, ``2`` for polarization.

    Examples
    --------
    >>> Tracer("Planck", 143.0, spin=2)
    Tracer(instrument='Planck', freq=143.0, spin=2)
    """

    instrument: str
    freq: float
    spin: int


@dataclass(frozen=True)
class BandInfo:
    """Per-band calibration info consumed by the dust model and field builder.

    Parameters
    ----------
    tracer : Tracer
        The tracer this band info describes.
    beam_fwhm_arcmin : float
        Gaussian beam FWHM in arcminutes; fed to ``hp.gauss_beam``.
    eff_freq_cmb : float or None, optional
        Effective (bandpass-weighted) frequency for CMB SED integration, GHz.
        Falls back to the nominal band center when ``None``.
    eff_freq_dust : float or None, optional
        Effective frequency for dust MBB integration, GHz. Falls back to the
        nominal band center when ``None``.
    unit_scale : float, optional
        Multiplicative scale applied to raw maps at load time (default ``1.0``;
        Planck/SO maps typically use ``1e6`` to convert K -> muK).
    """

    tracer: Tracer
    beam_fwhm_arcmin: float
    eff_freq_cmb: float | None = None
    eff_freq_dust: float | None = None
    unit_scale: float = 1.0


@dataclass(frozen=True)
class SpectrumKey:
    """Typed identifier for one (tracer1, tracer2, component) spectrum.

    Parameters
    ----------
    tracer1 : Tracer
        First tracer in the cross-spectrum.
    tracer2 : Tracer
        Second tracer. For auto-spectra, equal to ``tracer1``.
    comp : Comp
        Spectrum component (``"EE"``, ``"BB"``, ``"TE"``, ...).
    """

    tracer1: Tracer
    tracer2: Tracer
    comp: Comp
