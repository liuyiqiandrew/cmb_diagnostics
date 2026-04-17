"""Plot + save polarization-angle results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from cmb_diagnostics.estimators.base import FitResult


def save_npz(result: FitResult, path: str | Path) -> None:
    """Phase 5: thin wrapper over ``FitResult.save_npz`` with provenance."""
    raise NotImplementedError("Phase 5: delegate to FitResult.save_npz.")


def plot(
    result: FitResult,
    path: str | Path | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot polarization-angle sweep vs lmax cap. Returns (fig, ax).

    Phase 5: port from ``cmb_diagnoistics/PolAngEstimator.py`` plotting.
    """
    raise NotImplementedError(
        "Phase 5: port from cmb_diagnoistics/PolAngEstimator.py plotting "
        "(must return (fig, ax); save only when path is given)."
    )
