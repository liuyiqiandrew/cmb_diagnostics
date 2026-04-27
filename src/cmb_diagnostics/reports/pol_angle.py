"""Plot + save polarization-angle results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from cmb_diagnostics.estimators.base import FitResult


def save_npz(result: FitResult, path: str | Path) -> None:
    """Save a pol-angle :class:`FitResult` to an npz.

    Thin wrapper over :meth:`FitResult.save_npz`.

    Parameters
    ----------
    result : FitResult
        Pol-angle result.
    path : str or Path
        Destination.
    """
    result.save_npz(path)


def plot(
    result: FitResult,
    path: str | Path | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot a pol-angle sweep vs lmax cap with one series per pair.

    Parameters
    ----------
    result : FitResult
        Pol-angle result with ``values`` of shape ``(n_pairs, n_caps)``.
    path : str or Path or None, optional
        When given, save the figure there.
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw on; when ``None``, a new figure is created.
    **kwargs
        ``dpi`` is consumed for new-figure creation; other kwargs ignored.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure / axis pair. Y-axis is in degrees.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(dpi=kwargs.pop("dpi", 150))
    else:
        fig = ax.figure

    pairs = result.diagnostics.get("so_pairs")
    values_deg = result.values * 180.0 / np.pi
    errors_deg = result.errors * 180.0 / np.pi
    n_pairs = result.values.shape[0]
    for i in range(n_pairs):
        if pairs is not None:
            f1, f2 = pairs[i]
            label = f"s{int(f1)}xs{int(f2)}"
        else:
            label = f"pair_{i}"
        ax.errorbar(result.ell, values_deg[i], errors_deg[i], label=label, marker=".")

    ax.axhline(0, c="k", ls="--")
    ax.set_xlabel(r"Fitting limit $L_{up}$")
    ax.set_ylabel("Polarization angle (deg)")
    ax.legend()
    fig.tight_layout()

    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p)

    return fig, ax
