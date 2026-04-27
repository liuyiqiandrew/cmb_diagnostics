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
    """Save a TF :class:`FitResult` to an npz file.

    Thin wrapper over :meth:`FitResult.save_npz` exposed alongside
    :func:`plot` so callers can import from a single reporting module.

    Parameters
    ----------
    result : FitResult
        TF result.
    path : str or Path
        Destination.
    """
    result.save_npz(path)


def plot(
    results: Sequence[FitResult],
    path: str | Path | None = None,
    ax: Axes | None = None,
    reference_tf: Any = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot one or more TF results as errorbars.

    Parameters
    ----------
    results : sequence of FitResult
        One errorbar series per result; labels come from
        ``result.metadata['target']`` when present.
    path : str or Path or None, optional
        When given, save the figure there after plotting.
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw on; when ``None``, create a new figure.
    reference_tf : array-like or None, optional
        Overlay a dashed reference TF. Accepts either a ``(2, n)`` array or
        a ``(ell, tf)`` sequence.
    **kwargs
        ``dpi`` is consumed for new-figure creation; other kwargs are
        ignored.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure / axis pair.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(dpi=kwargs.pop("dpi", 150))
    else:
        fig = ax.figure

    for r in results:
        label = str(r.metadata.get("target", r.name))
        ax.errorbar(
            r.ell, r.values, r.errors,
            label=label, ls="", marker=".", alpha=0.5, capsize=3,
        )

    if reference_tf is not None:
        ref = reference_tf
        if hasattr(ref, "shape") and ref.ndim == 2 and ref.shape[0] == 2:
            ref_ell, ref_val = ref[0], ref[1]
        else:
            ref_ell, ref_val = ref
        ax.plot(ref_ell, ref_val, ls="--", c="k", label="reference")

    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel("Transfer function")
    ax.legend()
    fig.tight_layout()

    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p)

    return fig, ax


def plot_diagnostics(
    result: FitResult,
    out_dir: str | Path,
) -> Path | None:
    """Per-bin TF diagnostic plot: ``r``, ``dust_amp``, and chi^2 vs ell.

    Replaces the legacy per-bin ``debug_dust_fit_*.png`` /
    ``debug_tf_fit_*.png`` dumps to CWD.

    Parameters
    ----------
    result : FitResult
        TF result; ``diagnostics`` must contain at least one of ``"r"``,
        ``"dust_amp"``, ``"chi2_tf"`` for a plot to be produced.
    out_dir : str or Path
        Output directory; created if missing.

    Returns
    -------
    Path or None
        Path to the written PNG, or ``None`` when there are no diagnostics.
    """
    import matplotlib.pyplot as plt

    diag = result.diagnostics
    if not diag:
        return None

    keys = [k for k in ("r", "dust_amp", "chi2_tf") if k in diag]
    if not keys:
        return None

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(keys), 1, figsize=(6, 2.5 * len(keys)), dpi=150, sharex=True,
    )
    if len(keys) == 1:
        axes = [axes]

    ylabels = {"r": r"$r = \sqrt{\mathrm{TF}}$", "dust_amp": "dust amplitude", "chi2_tf": r"$\chi^2$ (TF fit)"}
    for ax, k in zip(axes, keys, strict=True):
        ax.plot(result.ell, diag[k], marker=".", ls="-", alpha=0.7)
        ax.set_ylabel(ylabels.get(k, k))
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel(r"$\ell$")
    fig.suptitle(result.name)
    fig.tight_layout()

    path = out_dir_p / f"{result.name}_diagnostics.png"
    fig.savefig(path)
    plt.close(fig)
    return path
