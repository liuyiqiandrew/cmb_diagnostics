"""Binned CMB reference container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from cmb_diagnostics._types import Comp

if TYPE_CHECKING:
    from cmb_diagnostics.models.bandpowers import Bandpowers


@dataclass(frozen=True)
class CMBReference:
    """Binned CMB reference spectra (from CAMB) keyed by component.

    Parameters
    ----------
    bandpowers : Bandpowers
        The binning used to produce ``cls_binned`` — kept so downstream code
        can reconstruct the effective ells.
    cls_binned : dict of Comp to numpy.ndarray, optional
        Component-keyed binned ``C_ell`` arrays aligned with
        ``bandpowers.effective_ell``. Typical keys: ``"TT"``, ``"EE"``,
        ``"BB"``, ``"TE"``.
    """

    bandpowers: Bandpowers
    cls_binned: dict[Comp, np.ndarray] = field(default_factory=dict)

    def get(self, comp: Comp) -> np.ndarray:
        """Return the binned ``C_ell`` for ``comp``.

        Parameters
        ----------
        comp : Comp
            Spectrum component (``"EE"``, ``"BB"``, ``"TE"``, ...).

        Returns
        -------
        numpy.ndarray
            Binned ``C_ell`` values.

        Raises
        ------
        KeyError
            When ``comp`` is absent from ``cls_binned``.
        """
        if comp not in self.cls_binned:
            raise KeyError(f"CMBReference has no {comp!r} spectrum")
        return self.cls_binned[comp]

    def has(self, comp: Comp) -> bool:
        """Return whether ``comp`` is populated.

        Parameters
        ----------
        comp : Comp
            Spectrum component to check.

        Returns
        -------
        bool
            ``True`` iff ``comp`` is present in ``cls_binned``.
        """
        return comp in self.cls_binned
