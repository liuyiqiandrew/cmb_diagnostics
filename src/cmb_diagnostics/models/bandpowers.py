"""Bandpower bin definition.

Replaces the scattered bins/e_l/bin_width/msk/e_dl2cl attributes on V1's
PSContainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from cmb_diagnostics.config import BandpowersConfig


@dataclass(frozen=True)
class Bandpowers:
    """Linear bandpower binning plus cached effective ell values.

    Parameters
    ----------
    nmt_bin : Any
        The underlying ``pymaster.NmtBin`` instance, or ``None`` for a
        binless object used only for the ``_effective_ell`` cache.
    bin_width : int
        Width in ell of each linear bin.
    lmin : int
        Lower ell cutoff for the :attr:`msk` mask.
    lmax : int
        Upper ell cutoff for the :attr:`msk` mask.
    is_dell : bool
        Whether ``nmt_bin`` bins ``D_ell`` (``True``) or ``C_ell``.
    _effective_ell : numpy.ndarray or None, optional
        Precomputed effective-ell array; avoids re-calling
        ``nmt_bin.get_effective_ells()``.
    """

    nmt_bin: Any
    bin_width: int
    lmin: int
    lmax: int
    is_dell: bool
    _effective_ell: np.ndarray | None = None

    @property
    def effective_ell(self) -> np.ndarray:
        """Effective ell per bin.

        Returns
        -------
        numpy.ndarray
            Effective ell values, prefering the cached ``_effective_ell`` over
            a live call to ``nmt_bin.get_effective_ells()``.

        Raises
        ------
        RuntimeError
            When neither the cache nor ``nmt_bin`` is available.
        """
        if self._effective_ell is not None:
            return self._effective_ell
        if self.nmt_bin is None:
            raise RuntimeError(
                "Bandpowers.effective_ell requires either nmt_bin or _effective_ell"
            )
        return self.nmt_bin.get_effective_ells()

    @property
    def msk(self) -> np.ndarray:
        """Boolean mask of bins strictly inside ``(lmin, lmax)``.

        Returns
        -------
        numpy.ndarray
            Boolean array aligned with :attr:`effective_ell`.
        """
        ell = self.effective_ell
        return (ell > self.lmin) & (ell < self.lmax)

    @property
    def dl2cl(self) -> np.ndarray:
        """Per-bin ``D_ell`` -> ``C_ell`` conversion factor.

        Returns
        -------
        numpy.ndarray
            ``2 * pi / (ell * (ell + 1))`` evaluated at :attr:`effective_ell`.
        """
        ell = self.effective_ell
        return 2 * np.pi / ell / (ell + 1)

    @classmethod
    def from_config(cls, cfg: BandpowersConfig, nside: int) -> Bandpowers:
        """Build a linearly-binned ``NmtBin`` plus effective-ell cache.

        Mirrors V1 ``PSContainer.init_nmt``.

        Parameters
        ----------
        cfg : BandpowersConfig
            Binning parameters (``bin_width``, ``lmin``, ``lmax``, ``is_Dell``).
        nside : int
            HEALPix resolution; fixes the maximum ell of the binning.

        Returns
        -------
        Bandpowers
            Populated :class:`Bandpowers` with ``nmt_bin`` and
            ``_effective_ell`` set.
        """
        import pymaster as nmt

        nmt_bin = nmt.NmtBin.from_nside_linear(nside, cfg.bin_width, is_Dell=cfg.is_Dell)
        eff_ell = nmt_bin.get_effective_ells()
        return cls(
            nmt_bin=nmt_bin,
            bin_width=cfg.bin_width,
            lmin=cfg.lmin,
            lmax=cfg.lmax,
            is_dell=cfg.is_Dell,
            _effective_ell=eff_ell,
        )
