"""Dataclass + container behavior: frozen, hashable, equal-by-value, round-trips."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from cmb_diagnostics import FieldSet, FitResult, Mask, Spectra, SpectrumKey, Tracer
from cmb_diagnostics._types import BandInfo


def test_tracer_frozen_and_hashable():
    t1 = Tracer("Planck", 143.0, 2)
    t2 = Tracer("Planck", 143.0, 2)
    assert t1 == t2
    assert hash(t1) == hash(t2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        t1.freq = 100.0  # type: ignore[misc]
    d = {t1: "ok"}
    assert d[t2] == "ok"


def test_bandinfo_default_fields():
    bi = BandInfo(Tracer("Planck", 143.0, 2), beam_fwhm_arcmin=7.27)
    assert bi.eff_freq_cmb is None
    assert bi.unit_scale == 1.0


def test_spectrumkey_equality_by_value():
    k1 = SpectrumKey(Tracer("Planck", 100.0, 2), Tracer("Planck", 143.0, 2), "EE")
    k2 = SpectrumKey(Tracer("Planck", 100.0, 2), Tracer("Planck", 143.0, 2), "EE")
    assert k1 == k2
    assert hash(k1) == hash(k2)


def test_mask_is_frozen():
    m = Mask(hp_map=np.ones(12), nside=1, fsky_effective=0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.nside = 2  # type: ignore[misc]


def test_spectra_add_get_roundtrip(spectrumkey_factory):
    s = Spectra()
    key = spectrumkey_factory()
    cl = np.arange(5, dtype=float)
    var = np.full(5, 0.1)
    s.add(key, cl, var)
    got_cl, got_var = s.get(key.tracer1, key.tracer2, key.comp)
    np.testing.assert_array_equal(got_cl, cl)
    np.testing.assert_array_equal(got_var, var)
    assert s.has(key.tracer1, key.tracer2, key.comp)
    assert not s.has(key.tracer1, key.tracer1, "EE")


def test_spectra_iter_pairs_filter(spectrumkey_factory):
    s = Spectra()
    k_ee = spectrumkey_factory(100.0, 143.0, "EE")
    k_bb = spectrumkey_factory(100.0, 143.0, "BB")
    s.add(k_ee, np.zeros(3))
    s.add(k_bb, np.zeros(3))
    ee_pairs = list(s.iter_pairs("EE"))
    assert len(ee_pairs) == 1
    assert ee_pairs[0] == (k_ee.tracer1, k_ee.tracer2)


def test_spectra_npz_roundtrip(tmp_path: Path, spectrumkey_factory):
    s = Spectra()
    k = spectrumkey_factory()
    cl = np.arange(5, dtype=float)
    s.add(k, cl, np.full(5, 0.2))
    p = tmp_path / "spec.npz"
    s.save_npz(p)
    s2 = Spectra.load_npz(p)
    cl2, var2 = s2.get(k.tracer1, k.tracer2, k.comp)
    np.testing.assert_array_equal(cl2, cl)
    np.testing.assert_array_equal(var2, np.full(5, 0.2))


def test_fieldset_add_get_tracers(tracer_factory):
    fs = FieldSet("planck")
    t0 = tracer_factory(spin=0)
    t2 = tracer_factory(spin=2)
    fs.add(t0, object())
    fs.add(t2, object())
    assert set(fs.tracers()) == {t0, t2}
    assert list(fs.tracers(spin=2)) == [t2]
    with pytest.raises(KeyError):
        fs.add(t0, object())


def test_fitresult_npz_roundtrip(tmp_path: Path):
    r = FitResult(
        name="demo",
        ell=np.arange(4, dtype=float),
        values=np.ones(4),
        errors=np.full(4, 0.1),
        diagnostics={"dust_amp": np.full(4, 2.0)},
    )
    p = tmp_path / "r.npz"
    r.save_npz(p)
    r2 = FitResult.load_npz(p)
    assert r2.name == "demo"
    np.testing.assert_array_equal(r2.values, r.values)
    np.testing.assert_array_equal(r2.diagnostics["dust_amp"], r.diagnostics["dust_amp"])
