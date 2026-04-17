"""CAMB reference loader (BBPower-style tables)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cmb_diagnostics.config import CambConfig
    from cmb_diagnostics.models.bandpowers import Bandpowers
    from cmb_diagnostics.models.cmb import CMBReference


def load_camb_reference(cfg: CambConfig, bandpowers: Bandpowers) -> CMBReference:
    """Parse a BBPower-style ``camb_lens_nobb.dat`` into a binned ``CMBReference``.

    Phase 3: port from V1 ``PSContainer.init_camb_dl``. Format notes: text file
    with rows ``[ell, TT, EE, BB, TE]`` in Dl starting at ell=2. Prepend one zero
    row to shift indexing to ell=0, slice to ``3*nside``, call
    ``bandpowers.nmt_bin.bin_cell(...)``, multiply by ``bandpowers.dl2cl`` to
    convert Dl->Cl.
    """
    raise NotImplementedError(
        "Phase 3: port from cmb_diagnoistics/PSContainer.py::PSContainer.init_camb_dl "
        "(bbpower_nobb parse + zero-row prepend + nmt_bin.bin_cell + dl2cl conversion)."
    )
