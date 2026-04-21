"""End-to-end regression against the V2 golden file ``test/bf_tf.npy``.

This test is marked ``slow``: it reads the full analysis mask and the SO SAT
polarization maps from Della (~GB-scale reads) and runs NaMaster MASTER.

The test skips cleanly when any required artefact is missing, so it is safe
to leave in the default test set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_BF = REPO_ROOT / "test" / "bf_tf.npy"
GOLDEN_ML = REPO_ROOT / "test" / "ml_tf.npy"
ISO_CONFIG = REPO_ROOT / "configs" / "iso_satp3.yaml"


pytestmark = pytest.mark.slow


def _require(paths: list[Path]) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"regression test requires Della data; missing: {missing}")


def _iso_paths_from_config(cfg) -> list[Path]:
    """Return the concrete file paths the ISO config references."""
    paths = [Path(cfg.camb.path), Path(cfg.mask.path)]
    for band in cfg.planck.bands:
        paths.append(Path(cfg.planck.map_template.format(freq=band.freq)))
    for band in cfg.so.bands:
        paths.append(Path(cfg.so.map_template.format(freq=band.freq)))
    return paths


def test_tf_ee_matches_golden_bf():
    """Run the new pipeline against the ISO config and compare to bf_tf.npy.

    Golden file convention (V1 ``TransferFuncEE.plot_and_save_transfer_func``):
    columns ``[ell, tf_90, var_90, tf_150, var_150]``.
    """
    pytest.importorskip("pygsm")
    pytest.importorskip("pymaster")

    if not ISO_CONFIG.exists():
        pytest.skip(f"config missing: {ISO_CONFIG}")

    from cmb_diagnostics import Config, Pipeline
    from cmb_diagnostics._types import Tracer

    cfg = Config.from_yaml(ISO_CONFIG)
    _require([GOLDEN_BF] + _iso_paths_from_config(cfg))

    pipe = Pipeline(cfg)
    pipe.load_mask()
    pipe.build_fieldsets()
    pipe.compute_spectra()
    tf_90 = pipe.estimate_tf_ee(target=Tracer("SO_SAT", 90.0, spin=2))
    tf_150 = pipe.estimate_tf_ee(target=Tracer("SO_SAT", 150.0, spin=2))

    golden = np.load(GOLDEN_BF)
    # Row shape depends on V1 column layout (ell, tf90, var90, tf150, var150).
    # Compare on the shared ell range. The new pipeline applies bp.msk, so
    # align by trimming the golden to the same ell values.
    ell_new = tf_90.ell
    ell_gold = golden[:, 0]
    common = np.isin(np.round(ell_gold), np.round(ell_new))
    assert common.any(), "no overlap between new and golden ell grids"
    g = golden[common]

    # β semantic change (1.53 -> 1.6) introduces a small numerical drift;
    # keep the tolerance loose until a V1-compatible mode is added.
    np.testing.assert_allclose(tf_90.values, g[:, 1], rtol=5e-2)
    np.testing.assert_allclose(tf_150.values, g[:, 3], rtol=5e-2)
