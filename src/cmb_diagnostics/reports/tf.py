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
    """Save a TF :class:`FitResult` to an npz via :meth:`FitResult.save_npz`."""
    result.save_npz(path)


def plot(
    results: Sequence[FitResult],
    path: str | Path | None = None,
    ax: Axes | None = None,
    reference_tf: Any = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot one or more TF results as errorbars.

    ``reference_tf`` (optional) is overlaid as a dashed black line. Pass either
    ``(ell, tf)`` or an array of shape ``(2, n)``. Writes to ``path`` only if
    given; always returns ``(fig, ax)``.
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
    """Per-bin TF diagnostic plot: r, dust amplitude, and fit chi2 vs ell.

    Replaces the legacy per-bin ``debug_dust_fit_*.png`` / ``debug_tf_fit_*.png``
    dumps to CWD. Writes ``{out_dir}/{result.name}_diagnostics.png`` and returns
    the path. Returns ``None`` if the result carries no diagnostics.
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
