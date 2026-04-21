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
    name: str
    ell: np.ndarray
    values: np.ndarray
    errors: np.ndarray
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save_npz(self, path: str | Path) -> None:
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

    def plot(
        self, ax: Axes | None = None, **kwargs: Any
    ) -> tuple[Figure, Axes]:
        """Plot the fit result.

        Dispatches on ``values.ndim``: 1-D → TF-style errorbars; 2-D →
        pol-angle sweep. Returns ``(fig, ax)``; never writes to disk.
        """
        if self.values.ndim == 2:
            from cmb_diagnostics.reports import pol_angle as _pa
            return _pa.plot(self, ax=ax, **kwargs)
        from cmb_diagnostics.reports import tf as _tf
        return _tf.plot([self], ax=ax, **kwargs)


@runtime_checkable
class Estimator(Protocol):
    def estimate(self, *args: Any, **kwargs: Any) -> FitResult: ...
