"""Config YAML + dict loading, validation, and round-trips."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cmb_diagnostics.config import Config, ConfigError


def test_from_yaml_loads_tiny_config(tiny_config: Config):
    assert tiny_config.schema_version == "1"
    assert tiny_config.nside == 16
    assert tiny_config.bandpowers.bin_width == 10
    assert tiny_config.bandpowers.is_Dell is True
    assert tiny_config.planck.name == "Planck"
    assert tiny_config.so.name == "SO_SAT"
    assert len(tiny_config.planck.bands) == 1
    assert tiny_config.planck.bands[0].freq == 143
    assert tiny_config.so.bands[0].beam_fwhm_arcmin == 27.4
    assert tiny_config.pol_angle.lmax_sweep == [30, 40]


def test_from_dict_matches_from_yaml(tiny_config_yaml: Path):
    data = yaml.safe_load(tiny_config_yaml.read_text())
    cfg_dict = Config.from_dict(data)
    cfg_yaml = Config.from_yaml(tiny_config_yaml)
    assert cfg_dict == cfg_yaml


def test_missing_schema_version_raises(tmp_path: Path):
    p = tmp_path / "no_schema.yaml"
    p.write_text("nside: 16\n")
    with pytest.raises(ConfigError, match="schema_version"):
        Config.from_yaml(p)


def test_wrong_schema_version_raises(tmp_path: Path):
    p = tmp_path / "wrong.yaml"
    p.write_text('schema_version: "99"\nnside: 16\n')
    with pytest.raises(ConfigError, match="schema_version"):
        Config.from_yaml(p)


def test_unknown_top_level_key_warns(tiny_config_yaml: Path):
    data = yaml.safe_load(tiny_config_yaml.read_text())
    data["frobnicate"] = True
    with pytest.warns(UserWarning, match="frobnicate"):
        Config.from_dict(data)


def test_duplicate_band_freq_raises(tiny_config_yaml: Path):
    data = yaml.safe_load(tiny_config_yaml.read_text())
    data["planck"]["bands"] = [
        {"freq": 143, "beam_fwhm_arcmin": 7.0},
        {"freq": 143, "beam_fwhm_arcmin": 9.0},
    ]
    with pytest.raises(ConfigError, match="unique"):
        Config.from_dict(data)


def test_missing_required_instrument_key_raises(tiny_config_yaml: Path):
    data = yaml.safe_load(tiny_config_yaml.read_text())
    del data["planck"]["map_template"]
    with pytest.raises(ConfigError, match="planck.map_template"):
        Config.from_dict(data)


def test_file_not_found_raises(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        Config.from_yaml(tmp_path / "missing.yaml")
