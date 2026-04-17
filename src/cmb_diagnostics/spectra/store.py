"""Spectra container keyed by SpectrumKey.

Pure container - no science. Replaces V1's many parallel dicts (planck_ee,
planck_xspec_ee_var, so_x_planck_ee, ...) and V2's Container.PSContainer.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from cmb_diagnostics._types import Comp, SpectrumKey, Tracer

if TYPE_CHECKING:
    from cmb_diagnostics.models.bandpowers import Bandpowers


class Spectra:
    """Power spectra + per-bin variance keyed by SpectrumKey."""

    def __init__(self, bandpowers: Bandpowers | None = None) -> None:
        self.bandpowers = bandpowers
        self.cls: dict[SpectrumKey, np.ndarray] = {}
        self.vars: dict[SpectrumKey, np.ndarray] = {}

    def add(
        self,
        key: SpectrumKey,
        cl: np.ndarray,
        var: np.ndarray | None = None,
    ) -> None:
        self.cls[key] = np.asarray(cl)
        if var is not None:
            self.vars[key] = np.asarray(var)

    def get(
        self, t1: Tracer, t2: Tracer, comp: Comp
    ) -> tuple[np.ndarray, np.ndarray | None]:
        key = SpectrumKey(t1, t2, comp)
        if key not in self.cls:
            raise KeyError(f"no spectrum for {key}")
        return self.cls[key], self.vars.get(key)

    def has(self, t1: Tracer, t2: Tracer, comp: Comp) -> bool:
        return SpectrumKey(t1, t2, comp) in self.cls

    def iter_pairs(self, comp: Comp) -> Iterator[tuple[Tracer, Tracer]]:
        for key in self.cls:
            if key.comp == comp:
                yield key.tracer1, key.tracer2

    def keys(self) -> Iterable[SpectrumKey]:
        return self.cls.keys()

    def save_npz(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ordered = list(self.cls.keys())
        payload: dict[str, np.ndarray] = {}
        meta: list[tuple[str, float, int, str, float, int, str]] = []
        for i, k in enumerate(ordered):
            payload[f"cl_{i}"] = self.cls[k]
            if k in self.vars:
                payload[f"var_{i}"] = self.vars[k]
            meta.append(
                (
                    k.tracer1.instrument,
                    k.tracer1.freq,
                    k.tracer1.spin,
                    k.tracer2.instrument,
                    k.tracer2.freq,
                    k.tracer2.spin,
                    k.comp,
                )
            )
        payload["_meta"] = np.array(meta, dtype=object)
        np.savez(p, **payload)

    @classmethod
    def load_npz(cls, path: str | Path) -> Spectra:
        data = np.load(Path(path), allow_pickle=True)
        meta = data["_meta"]
        spec = cls()
        for i, row in enumerate(meta):
            t1 = Tracer(str(row[0]), float(row[1]), int(row[2]))
            t2 = Tracer(str(row[3]), float(row[4]), int(row[5]))
            key = SpectrumKey(t1, t2, str(row[6]))
            cl = data[f"cl_{i}"]
            var = data[f"var_{i}"] if f"var_{i}" in data.files else None
            spec.add(key, cl, var)
        return spec

    def __repr__(self) -> str:
        return f"Spectra(nkeys={len(self.cls)}, bandpowers={self.bandpowers!r})"

    def __len__(self) -> int:
        return len(self.cls)
