"""FieldSet: NaMaster fields for one instrument view of the sky.

Replaces V1's twin planck_f0/planck_f2/so_f0/so_f2 dicts on PSContainer, and
V2's NmtFieldContainer (which this class is close to, but keyed on Tracer
instead of bare frequency).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

from cmb_diagnostics._types import Tracer

if TYPE_CHECKING:
    import pymaster as nmt  # noqa: F401


class FieldSet:
    """Dict-like container of NaMaster fields keyed by :class:`Tracer`.

    Replaces V1's twin ``planck_f0``/``planck_f2``/``so_f0``/``so_f2`` dicts and
    V2's ``NmtFieldContainer`` (which this class is close to, but keyed on
    :class:`Tracer` instead of bare frequency).

    Parameters
    ----------
    name : str
        Human-readable label (typically the instrument name).
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.fields: dict[Tracer, Any] = {}

    def add(self, tracer: Tracer, field: Any) -> None:
        """Register ``field`` under ``tracer``.

        Parameters
        ----------
        tracer : Tracer
            Identifier.
        field : Any
            A ``pymaster.NmtField`` (or a drop-in mock for tests).

        Raises
        ------
        KeyError
            If ``tracer`` is already registered.
        """
        if tracer in self.fields:
            raise KeyError(f"{tracer} already has a field in FieldSet({self.name!r})")
        self.fields[tracer] = field

    def get(self, tracer: Tracer) -> Any:
        """Return the field registered under ``tracer``.

        Parameters
        ----------
        tracer : Tracer
            Identifier.

        Returns
        -------
        Any
            The stored field.

        Raises
        ------
        KeyError
            When ``tracer`` has no field.
        """
        if tracer not in self.fields:
            raise KeyError(f"no field for {tracer} in FieldSet({self.name!r})")
        return self.fields[tracer]

    def tracers(self, spin: int | None = None) -> Iterable[Tracer]:
        """Return stored tracers, optionally filtered by spin.

        Parameters
        ----------
        spin : int or None, optional
            When given, return only tracers whose ``spin`` matches.

        Returns
        -------
        list of Tracer
            Matching tracers in insertion order.
        """
        if spin is None:
            return list(self.fields.keys())
        return [t for t in self.fields if t.spin == spin]

    def __contains__(self, tracer: Tracer) -> bool:
        return tracer in self.fields

    def __iter__(self) -> Iterator[Tracer]:
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def __repr__(self) -> str:
        return f"FieldSet(name={self.name!r}, ntracers={len(self.fields)})"
