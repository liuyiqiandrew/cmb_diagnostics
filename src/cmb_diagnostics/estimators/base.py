"""FitResult dataclass + Estimator Protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class FitResult:
    """Result of a single estimator run (TF, pol-angle, ...).

    Parameters
    ----------
    name : str
        Stable identifier used as the default filename stem.
    ell : numpy.ndarray
        Per-bin ell axis aligned with ``values``.
    values : numpy.ndarray
        Fit values. 1-D for TF, 2-D ``(n_pairs, n_caps)`` for pol-angle.
    errors : numpy.ndarray
        1-sigma errors aligned with ``values``.
    diagnostics : dict of str to numpy.ndarray, optional
        Per-bin quantities useful for debugging (``"r"``, ``"dust_amp"``,
        ``"chi2_tf"``, ``"so_pairs"``, ...). Serialized to ``diag_<key>``
        when saved.
    metadata : dict of str to Any, optional
        Freeform metadata (e.g. the estimator name or target :class:`Tracer`).
        Not serialized.
    """

    name: str
    ell: np.ndarray
    values: np.ndarray
    errors: np.ndarray
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save_npz(self, path: str | Path) -> None:
        """Serialize this result to a single npz file.

        Parameters
        ----------
        path : str or Path
            Destination path; parent directories are created as needed.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            p,
            name=np.array(self.name),
            ell=self.ell,
            values=self.values,
            errors=self.errors,
            **{f"diag_{k}": v for k, v in self.diagnostics.items()},
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> FitResult:
        """Reconstruct a :class:`FitResult` written by :meth:`save_npz`.

        Parameters
        ----------
        path : str or Path
            Source npz path.

        Returns
        -------
        FitResult
            Populated result; ``metadata`` is empty (not round-tripped).
        """
        p = Path(path)
        data = np.load(p, allow_pickle=False)
        diagnostics = {
            k[len("diag_"):]: data[k] for k in data.files if k.startswith("diag_")
        }
        return cls(
            name=str(data["name"].item()),
            ell=data["ell"],
            values=data["values"],
            errors=data["errors"],
            diagnostics=diagnostics,
            metadata={},
        )

    def __repr__(self) -> str:
        n = len(self.values)
        if n == 0:
            return f"FitResult(name={self.name!r}, nbins=0)"
        med = float(np.median(self.values))
        err = float(np.median(self.errors))
        return f"FitResult(name={self.name!r}, nbins={n}, median={med:.3g}\u00b1{err:.2g})"

    def _repr_html_(self) -> str:
        if self.values.ndim == 2:
            return self._repr_html_2d()
        rows = [
            f"<tr><td>{ell:g}</td><td>{v:.4g}</td><td>{e:.3g}</td></tr>"
            for ell, v, e in zip(self.ell, self.values, self.errors, strict=True)
        ]
        return (
            f"<table><caption><b>FitResult</b>: {self.name} (nbins={len(self.values)})"
            "</caption>"
            "<thead><tr><th>ell</th><th>value</th><th>error</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    def _repr_html_2d(self) -> str:
        pair_labels = self.diagnostics.get("so_pairs")
        n_rows = self.values.shape[0]
        if pair_labels is not None and len(pair_labels) == n_rows:
            row_names = [f"{int(p[0])}x{int(p[1])}" for p in pair_labels]
        else:
            row_names = [str(i) for i in range(n_rows)]
        cells = []
        for i, name in enumerate(row_names):
            for j, ell in enumerate(self.ell):
                cells.append(
                    f"<tr><td>{name}</td><td>{ell:g}</td>"
                    f"<td>{self.values[i, j]:.4g}</td><td>{self.errors[i, j]:.3g}</td></tr>"
                )
        return (
            f"<table><caption><b>FitResult</b>: {self.name} "
            f"(shape={self.values.shape})</caption>"
            "<thead><tr><th>pair</th><th>ell_cap</th><th>value</th><th>error</th></tr></thead>"
            f"<tbody>{''.join(cells)}</tbody></table>"
        )

    def plot(
        self, ax: Axes | None = None, **kwargs: Any
    ) -> tuple[Figure, Axes]:
        """Plot the fit result.

        Dispatches on ``values.ndim``: 1-D renders TF-style errorbars; 2-D
        renders a pol-angle sweep. Never writes to disk.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            Axis to draw on; when ``None``, a new figure is created.
        **kwargs
            Forwarded to the delegated plotter.

        Returns
        -------
        tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
            The figure / axis pair.
        """
        if self.values.ndim == 2:
            from cmb_diagnostics.reports import pol_angle as _pa
            return _pa.plot(self, ax=ax, **kwargs)
        from cmb_diagnostics.reports import tf as _tf
        return _tf.plot([self], ax=ax, **kwargs)


@runtime_checkable
class Estimator(Protocol):
    """Structural protocol for any estimator returning a :class:`FitResult`.

    Implementations need only expose an ``estimate`` method; the concrete
    signature is left to the implementer (e.g. ``estimate(target=...)`` for
    transfer-function estimators, ``estimate()`` for the pol-angle sweep).
    """

    def estimate(self, *args: Any, **kwargs: Any) -> FitResult:
        """Run the estimator and return its :class:`FitResult`."""
        ...
