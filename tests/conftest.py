"""Shared pytest fixtures for the refactor test suite."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

import cmb_diagnostics
from cmb_diagnostics._types import SpectrumKey, Tracer
from cmb_diagnostics.config import Config
from cmb_diagnostics.models.bandpowers import Bandpowers


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def assert_new_package_on_sys_path(repo_root: Path) -> None:
    """Guarantees tests exercise src/cmb_diagnostics, not the legacy repo-root package.

    If a contributor has PYTHONPATH=<parent_of_repo>, Python resolves ``import
    cmb_diagnostics`` to the legacy re-export and silently shadows the new code.
    Fail the whole suite loudly instead of having tests pass against V1.
    """
    pkg_path = Path(cmb_diagnostics.__file__).resolve()
    expected = repo_root / "src" / "cmb_diagnostics" / "__init__.py"
    if pkg_path != expected:
        raise RuntimeError(
            "cmb_diagnostics resolves to\n"
            f"  {pkg_path}\n"
            f"expected {expected}\n"
            "Most likely your PYTHONPATH contains the repo's parent directory; unset it "
            "and re-run `pip install -e \".[dev]\"` before testing."
        )


@pytest.fixture
def tiny_config_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.yaml"
    path.write_text(
        dedent(
            """
            schema_version: "1"
            output_dir: {out}
            nside: 16

            bandpowers:
              bin_width: 10
              lmin: 20
              lmax: 40
              is_Dell: true

            mask:
              kind: boxes
              apodize: false
              threshold: 1.0e-5
              boxes:
                - [[-30, -20], [30, 20]]

            camb:
              path: /nonexistent/camb_lens_nobb.dat
              format: bbpower_nobb

            dust:
              beta: 1.6
              Td_kelvin: 19.6
              nu0_ghz: 353.0

            planck:
              name: Planck
              map_template: /nonexistent/planck_{{freq}}.fits
              unit_scale: 1.0e6
              pixelization: healpix_equatorial
              purify_e: true
              purify_b: true
              bands:
                - {{freq: 143, beam_fwhm_arcmin: 7.27, eff_freq_cmb: 142.7, eff_freq_dust: 148.2}}

            so:
              name: SO_SAT
              map_template: /nonexistent/so_f{{freq:03d}}.fits
              unit_scale: 1.0e6
              pixelization: car
              purify_e: false
              purify_b: false
              bands:
                - {{freq: 90, beam_fwhm_arcmin: 27.4}}

            pol_angle:
              lmin: 20
              lmax_sweep: [30, 40]
            """
        ).format(out=tmp_path / "out").strip()
    )
    return path


@pytest.fixture
def tiny_config(tiny_config_yaml: Path) -> Config:
    return Config.from_yaml(tiny_config_yaml)


@pytest.fixture
def synthetic_mask(nside_small: int = 16) -> np.ndarray:
    npix = 12 * nside_small * nside_small
    m = np.zeros(npix, dtype=np.float64)
    m[: npix // 5] = 1.0
    return m


@pytest.fixture
def synthetic_camb_dat(tmp_path: Path) -> Path:
    path = tmp_path / "camb_lens_nobb.dat"
    lines = []
    for ell in range(2, 101):
        tt = 1000.0 / ell
        ee = 10.0 / ell
        bb = 0.0
        te = 50.0 / ell
        lines.append(f"{ell} {tt:.4e} {ee:.4e} {bb:.4e} {te:.4e}")
    path.write_text("\n".join(lines))
    return path


@pytest.fixture
def tracer_factory() -> Callable[..., Tracer]:
    def make(instrument: str = "Planck", freq: float = 143.0, spin: int = 2) -> Tracer:
        return Tracer(instrument=instrument, freq=freq, spin=spin)
    return make


@pytest.fixture
def bandpowers_factory() -> Callable[..., Bandpowers]:
    def make(nbins: int = 5, bin_width: int = 20, lmin: int = 30, lmax: int = 300) -> Bandpowers:
        effective_ell = np.linspace(lmin + bin_width / 2, lmax - bin_width / 2, nbins)
        return Bandpowers(
            nmt_bin=None,
            bin_width=bin_width,
            lmin=lmin,
            lmax=lmax,
            is_dell=True,
            _effective_ell=effective_ell,
        )
    return make


@pytest.fixture
def spectrumkey_factory(tracer_factory: Callable[..., Tracer]) -> Callable[..., SpectrumKey]:
    def make(freq1: float = 100.0, freq2: float = 143.0, comp: str = "EE") -> SpectrumKey:
        t1 = tracer_factory(freq=freq1)
        t2 = tracer_factory(freq=freq2)
        return SpectrumKey(tracer1=t1, tracer2=t2, comp=comp)
    return make


@pytest.fixture(scope="session")
def golden_tf_bf(repo_root: Path) -> np.ndarray:
    path = repo_root / "test" / "bf_tf.npy"
    if not path.exists():
        pytest.skip(f"golden file missing: {path}")
    return np.load(path)


@pytest.fixture(scope="session")
def golden_tf_ml(repo_root: Path) -> np.ndarray:
    path = repo_root / "test" / "ml_tf.npy"
    if not path.exists():
        pytest.skip(f"golden file missing: {path}")
    return np.load(path)
