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
    """Parse a BBPower-style CAMB ``.dat`` into a binned :class:`CMBReference`.

    Format (BBPower-style): rows ``[ell, TT, EE, BB, TE]`` in ``D_ell``
    starting at ``ell = 1``. The loader prepends one zero row to shift indexing
    to ``ell = 0``, slices to ``3 * nside``, converts per multipole from
    ``D_ell`` to ``C_ell`` (``C_ell = D_ell * 2*pi / (ell*(ell+1))``), then
    delegates the per-bin average to ``nmt_bin.bin_cell``. When the bin was
    built with ``is_Dell=True`` (i.e. ``bin_cell`` returns ``<D_ell>_bin``),
    we multiply by ``bandpowers.dl2cl`` to recover ``<C_ell>_bin``; when
    ``is_Dell=False`` the binned output is already ``<C_ell>_bin``.

    Parameters
    ----------
    cfg : CambConfig
        CAMB loader config; only ``format == "bbpower_nobb"`` is supported.
    bandpowers : Bandpowers
        Bin definition; must expose a non-None ``nmt_bin``.
    nside : int
        HEALPix resolution; sets the ``lmax = 3 * nside`` slice before binning.

    Returns
    -------
    CMBReference
        Container populated with ``TT``, ``EE``, ``BB``, ``TE`` binned
        ``C_ell`` arrays.

    Raises
    ------
    ValueError
        When ``cfg.format`` is unrecognized.
    RuntimeError
        When ``bandpowers.nmt_bin`` is ``None``.
    """
    if cfg.format != "bbpower_nobb":
        raise ValueError(f"unsupported camb format: {cfg.format!r}")
    if bandpowers.nmt_bin is None:
        raise RuntimeError("load_camb_reference requires bandpowers.nmt_bin")

    dl = np.loadtxt(cfg.path)
    dl = np.concatenate([np.zeros((1, dl.shape[1]), dtype=dl.dtype), dl], axis=0)
    lmax = nside * 3

    # Per-ell D_ell -> C_ell so bin_cell sees C_ell. Without this, an
    # is_Dell=True bin re-applies ell(ell+1)/(2pi) to a D_ell input and
    # the post-bin dl2cl multiply only cancels one factor, leaving
    # <D_ell>_bin where <C_ell>_bin was intended.
    ells = np.arange(lmax)
    inv_prefactor = np.zeros(lmax, dtype=float)
    inv_prefactor[1:] = 2 * np.pi / (ells[1:] * (ells[1:] + 1))
    cl_table = dl[:lmax, :] * inv_prefactor[:, None]

    # bin_cell with is_Dell=True returns <D_ell>_bin from a C_ell input;
    # multiply by dl2cl to land back at <C_ell>_bin. is_Dell=False already
    # yields <C_ell>_bin, so no rescale.
    post = bandpowers.dl2cl if bandpowers.is_dell else 1.0

    ee = bandpowers.nmt_bin.bin_cell(cl_table[:, 2]) * post
    bb = bandpowers.nmt_bin.bin_cell(cl_table[:, 3]) * post
    te = bandpowers.nmt_bin.bin_cell(cl_table[:, 4]) * post
    tt = bandpowers.nmt_bin.bin_cell(cl_table[:, 1]) * post

    return CMBReference(
        bandpowers=bandpowers,
        cls_binned={"TT": tt, "EE": ee, "BB": bb, "TE": te},
    )
