"""Typed configuration + YAML/dict loader.

See docs/configuration.md for the full schema.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, is_dataclass
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a Config document fails validation."""


SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class BandpowersConfig:
    bin_width: int
    lmin: int
    lmax: int
    is_Dell: bool = True


@dataclass(frozen=True)
class MaskConfig:
    kind: str
    path: str | None = None
    pixelization: str = "healpix"
    apodize: bool = True
    threshold: float = 1e-5
    boxes: list[list[list[float]]] | None = None


@dataclass(frozen=True)
class CambConfig:
    path: str
    format: str = "bbpower_nobb"


@dataclass(frozen=True)
class DustConfig:
    beta: float = 1.6
    Td_kelvin: float = 19.6
    nu0_ghz: float = 353.0


@dataclass(frozen=True)
class BandConfig:
    freq: float
    beam_fwhm_arcmin: float
    eff_freq_cmb: float | None = None
    eff_freq_dust: float | None = None


@dataclass(frozen=True)
class InstrumentConfig:
    name: str
    map_template: str
    unit_scale: float = 1e6
    pixelization: str = "healpix_equatorial"
    purify_e: bool = False
    purify_b: bool = False
    bands: list[BandConfig] = field(default_factory=list)


@dataclass(frozen=True)
class PolAngleConfig:
    lmin: int = 30
    lmax_sweep: list[int] = field(default_factory=lambda: [200, 250, 300, 350, 400, 450, 500])


@dataclass(frozen=True)
class AdvancedConfig:
    fisher_error_form: str = "v1"
    write_diagnostic_plots: bool = False


@dataclass(frozen=True)
class Config:
    schema_version: str
    output_dir: Path
    nside: int
    bandpowers: BandpowersConfig
    mask: MaskConfig
    camb: CambConfig
    dust: DustConfig
    planck: InstrumentConfig
    so: InstrumentConfig
    pol_angle: PolAngleConfig
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"config file not found: {p}")
        with p.open("r") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ConfigError(f"top-level YAML must be a mapping, got {type(data).__name__}")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        return _build_config(data)


_KNOWN_SECTIONS = {
    "schema_version",
    "output_dir",
    "nside",
    "bandpowers",
    "mask",
    "camb",
    "dust",
    "planck",
    "so",
    "pol_angle",
    "advanced",
}


def _require(d: dict[str, Any], key: str, path: str) -> Any:
    if key not in d:
        raise ConfigError(f"{path}.{key}: required")
    return d[key]


def _build_config(data: dict[str, Any]) -> Config:
    unknown = set(data.keys()) - _KNOWN_SECTIONS
    if unknown:
        warnings.warn(
            f"Unknown top-level config keys ignored: {sorted(unknown)}",
            stacklevel=3,
        )

    schema_version = str(_require(data, "schema_version", ""))
    if schema_version != SCHEMA_VERSION:
        raise ConfigError(
            f"schema_version: expected {SCHEMA_VERSION!r}, got {schema_version!r}"
        )

    output_dir = Path(_require(data, "output_dir", ""))
    nside = int(_require(data, "nside", ""))

    bp = _build_from_mapping(
        _require(data, "bandpowers", ""), BandpowersConfig, "bandpowers"
    )
    mask = _build_from_mapping(_require(data, "mask", ""), MaskConfig, "mask")
    camb = _build_from_mapping(_require(data, "camb", ""), CambConfig, "camb")
    dust = _build_from_mapping(data.get("dust", {}), DustConfig, "dust")
    planck = _build_instrument(_require(data, "planck", ""), "planck")
    so = _build_instrument(_require(data, "so", ""), "so")
    pol_angle = _build_from_mapping(
        data.get("pol_angle", {}), PolAngleConfig, "pol_angle"
    )
    advanced = _build_from_mapping(data.get("advanced", {}), AdvancedConfig, "advanced")

    return Config(
        schema_version=schema_version,
        output_dir=output_dir,
        nside=nside,
        bandpowers=bp,
        mask=mask,
        camb=camb,
        dust=dust,
        planck=planck,
        so=so,
        pol_angle=pol_angle,
        advanced=advanced,
    )


def _build_instrument(data: dict[str, Any], path: str) -> InstrumentConfig:
    known = {f.name for f in dc_fields(InstrumentConfig)}
    unknown = set(data.keys()) - known
    if unknown:
        warnings.warn(
            f"Unknown keys under {path}: {sorted(unknown)}", stacklevel=4
        )
    raw_bands = _require(data, "bands", path)
    if not isinstance(raw_bands, list):
        raise ConfigError(f"{path}.bands: expected list, got {type(raw_bands).__name__}")
    bands: list[BandConfig] = []
    for i, raw in enumerate(raw_bands):
        bands.append(_build_from_mapping(raw, BandConfig, f"{path}.bands[{i}]"))
    freqs = [b.freq for b in bands]
    if len(set(freqs)) != len(freqs):
        raise ConfigError(f"{path}.bands[*].freq must be unique, got {freqs}")
    return InstrumentConfig(
        name=_require(data, "name", path),
        map_template=_require(data, "map_template", path),
        unit_scale=float(data.get("unit_scale", 1e6)),
        pixelization=str(data.get("pixelization", "healpix_equatorial")),
        purify_e=bool(data.get("purify_e", False)),
        purify_b=bool(data.get("purify_b", False)),
        bands=bands,
    )


def _build_from_mapping(data: Any, cls: type, path: str) -> Any:
    from dataclasses import MISSING

    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected mapping, got {type(data).__name__}")
    known = {f.name for f in dc_fields(cls)}
    unknown = set(data.keys()) - known
    if unknown:
        warnings.warn(f"Unknown keys under {path}: {sorted(unknown)}", stacklevel=4)
    kwargs: dict[str, Any] = {}
    for f in dc_fields(cls):
        if f.name in data:
            kwargs[f.name] = data[f.name]
            continue
        required = f.default is MISSING and f.default_factory is MISSING
        if required:
            raise ConfigError(f"{path}.{f.name}: required")
    return cls(**kwargs)
