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
    nmt_bin: Any
    bin_width: int
    lmin: int
    lmax: int
    is_dell: bool
    _effective_ell: np.ndarray | None = None

    @property
    def effective_ell(self) -> np.ndarray:
        if self._effective_ell is not None:
            return self._effective_ell
        if self.nmt_bin is None:
            raise RuntimeError(
                "Bandpowers.effective_ell requires either nmt_bin or _effective_ell"
            )
        return self.nmt_bin.get_effective_ells()

    @property
    def msk(self) -> np.ndarray:
        ell = self.effective_ell
        return (ell > self.lmin) & (ell < self.lmax)

    @property
    def dl2cl(self) -> np.ndarray:
        ell = self.effective_ell
        return 2 * np.pi / ell / (ell + 1)

    @classmethod
    def from_config(cls, cfg: BandpowersConfig, nside: int) -> Bandpowers:
        """Build a linearly-binned NmtBin + effective-ell cache.

        Mirrors V1 ``PSContainer.init_nmt``.
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
