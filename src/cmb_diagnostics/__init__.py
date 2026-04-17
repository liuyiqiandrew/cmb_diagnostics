"""Public API for the cmb_diagnostics refactor.

All 15 stable names listed in docs/api.md are re-exported at the top level.
Submodules (io, spectra, models, estimators, fields, fitting, reports) are
usable; internals prefixed with ``_`` are not.
"""

from __future__ import annotations

from cmb_diagnostics._types import BandInfo, SpectrumKey, Tracer
from cmb_diagnostics.config import Config
from cmb_diagnostics.estimators.base import FitResult
from cmb_diagnostics.estimators.pol_angle import PolarizationAngleEB
from cmb_diagnostics.estimators.transfer_function import (
    TransferFunctionEE,
    TransferFunctionTE,
)
from cmb_diagnostics.fields.container import FieldSet
from cmb_diagnostics.io.masks import Mask
from cmb_diagnostics.models.bandpowers import Bandpowers
from cmb_diagnostics.models.cmb import CMBReference
from cmb_diagnostics.models.dust import MBBDustModel
from cmb_diagnostics.pipeline import Pipeline
from cmb_diagnostics.spectra.store import Spectra

__all__ = [
    "BandInfo",
    "Bandpowers",
    "CMBReference",
    "Config",
    "FieldSet",
    "FitResult",
    "MBBDustModel",
    "Mask",
    "Pipeline",
    "PolarizationAngleEB",
    "Spectra",
    "SpectrumKey",
    "Tracer",
    "TransferFunctionEE",
    "TransferFunctionTE",
]
