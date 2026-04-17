"""Identifier dataclasses shared across the package.

Kept in one module so every submodule has a single import source and so circular
imports between `fields`, `spectra`, `models`, and `estimators` cannot arise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Comp = Literal["EE", "EB", "BE", "BB", "TE", "TB", "TT"]


@dataclass(frozen=True)
class Tracer:
    instrument: str
    freq: float
    spin: int


@dataclass(frozen=True)
class BandInfo:
    tracer: Tracer
    beam_fwhm_arcmin: float
    eff_freq_cmb: float | None = None
    eff_freq_dust: float | None = None
    unit_scale: float = 1.0


@dataclass(frozen=True)
class SpectrumKey:
    tracer1: Tracer
    tracer2: Tracer
    comp: Comp
