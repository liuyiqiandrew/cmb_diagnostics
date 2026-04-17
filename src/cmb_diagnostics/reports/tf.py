"""Plot + save transfer-function results."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from cmb_diagnostics.estimators.base import FitResult


def save_npz(result: FitResult, path: str | Path) -> None:
    """Save a TF :class:`FitResult` to an npz.

    Phase 5: thin wrapper over ``FitResult.save_npz``; may add provenance
    metadata.
    """
    raise NotImplementedError(
        "Phase 5: delegate to FitResult.save_npz + attach run provenance."
    )


def plot(
    results: Sequence[FitResult],
    path: str | Path | None = None,
    ax: Axes | None = None,
    reference_tf: Path | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot one or more TF results. Returns (fig, ax); writes to ``path`` if given.

    Phase 5: port from V1
    ``TransferFuncEstimator.TransferFuncEE.plot_and_save_transfer_func`` and
    ``test/tf_plotter.py`` (which loads bf_tf.npy + ml_tf.npy + reference_tf).
    """
    raise NotImplementedError(
        "Phase 5: port from cmb_diagnoistics/TransferFuncEstimator.py::"
        "TransferFuncEE.plot_and_save_transfer_func and test/tf_plotter.py "
        "(must return (fig, ax); save only when path is given)."
    )


def plot_diagnostics(
    result: FitResult,
    out_dir: str | Path,
) -> None:
    """Write per-bin dust-fit / TF-fit diagnostic PNGs under ``out_dir``.

    Opt-in only; called by CLI/Pipeline when
    ``cfg.advanced.write_diagnostic_plots`` is true. Replaces the V2
    side-effect writes of ``debug_dust_fit_*.png`` / ``debug_tf_fit_*.png``
    to CWD.

    Phase 5: port logic from ``cmb_diagnoistics/Estimator.py::SOPlkTF.__tf_ee``
    per-bin debug plots.
    """
    raise NotImplementedError(
        "Phase 5: port per-bin debug plots from "
        "cmb_diagnoistics/Estimator.py::SOPlkTF.__tf_ee "
        "(write under out_dir, not CWD)."
    )
