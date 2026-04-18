"""CAMB reference loader (BBPower-style tables)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cmb_diagnostics.models.cmb import CMBReference

if TYPE_CHECKING:
    from cmb_diagnostics.config import CambConfig
    from cmb_diagnostics.models.bandpowers import Bandpowers


def load_camb_reference(
    cfg: CambConfig,
    bandpowers: Bandpowers,
    nside: int,
) -> CMBReference:
    """Parse ``camb_lens_nobb.dat`` into a binned ``CMBReference``.

    Format (BBPower-style): rows ``[ell, TT, EE, BB, TE]`` in Dl starting at
    ell=2. Prepend one zero row to shift indexing to ell=0, slice to ``3*nside``,
    ``nmt_bin.bin_cell`` to bandpowers, multiply by ``bandpowers.dl2cl`` to
    convert Dl -> Cl. Ported from V1 ``PSContainer.init_camb_dl``.
    """
    if cfg.format != "bbpower_nobb":
        raise ValueError(f"unsupported camb format: {cfg.format!r}")
    if bandpowers.nmt_bin is None:
        raise RuntimeError("load_camb_reference requires bandpowers.nmt_bin")

    dl = np.loadtxt(cfg.path)
    dl = np.concatenate([np.zeros((1, dl.shape[1]), dtype=dl.dtype), dl], axis=0)
    lmax = nside * 3
    dl2cl = bandpowers.dl2cl

    ee = bandpowers.nmt_bin.bin_cell(dl[:lmax, 2]) * dl2cl
    bb = bandpowers.nmt_bin.bin_cell(dl[:lmax, 3]) * dl2cl
    te = bandpowers.nmt_bin.bin_cell(dl[:lmax, 4]) * dl2cl
    tt = bandpowers.nmt_bin.bin_cell(dl[:lmax, 1]) * dl2cl

    return CMBReference(
        bandpowers=bandpowers,
        cls_binned={"TT": tt, "EE": ee, "BB": bb, "TE": te},
    )
