"""Typed configuration + YAML/dict loader.

See ``docs/configuration.md`` for the full schema.
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
    """Linear bandpower binning parameters.

    Parameters
    ----------
    bin_width : int
        Width in ell of each linear bin.
    lmin : int
        Lower ell cutoff; bins with effective ell at or below this value are
        dropped from downstream fits.
    lmax : int
        Upper ell cutoff.
    is_Dell : bool, optional
        Whether ``nmt.NmtBin`` bins ``D_ell`` (``True``, default) or ``C_ell``.
        The pipeline divides by ``ell(ell+1)/(2*pi)`` downstream when ``True``.
    """

    bin_width: int
    lmin: int
    lmax: int
    is_Dell: bool = True


@dataclass(frozen=True)
class MaskConfig:
    """Mask construction parameters.

    Parameters
    ----------
    kind : str
        One of ``"file"`` (read ``path``) or ``"boxes"`` (OR of lat/lon boxes).
    path : str or None, optional
        FITS mask path, required when ``kind == "file"``.
    pixelization : str, optional
        ``"healpix"`` (default) or ``"car"`` — dispatches the file loader.
    apodize : bool, optional
        If ``True``, binarize the input mask and run the C2 apodization
        pipeline. If ``False``, preserve input weights and zero sub-threshold
        pixels in place.
    threshold : float, optional
        Threshold used to binarize (``apodize=True``) or zero (``apodize=False``)
        the raw mask. Default ``1e-5``.
    boxes : list of list of list of float or None, optional
        For ``kind == "boxes"``: list of ``[[dec_min, ra_min], [dec_max, ra_max]]``
        rectangles in degrees.
    """

    kind: str
    path: str | None = None
    pixelization: str = "healpix"
    apodize: bool = True
    threshold: float = 1e-5
    boxes: list[list[list[float]]] | None = None


@dataclass(frozen=True)
class CambConfig:
    """CAMB reference-spectrum loader parameters.

    Parameters
    ----------
    path : str
        Path to the CAMB ``.dat`` file (BBPower-style rows:
        ``[ell, TT, EE, BB, TE]`` in D_ell).
    format : str, optional
        Table format identifier. Only ``"bbpower_nobb"`` is currently
        recognized.
    """

    path: str
    format: str = "bbpower_nobb"


@dataclass(frozen=True)
class DustConfig:
    """Modified-blackbody dust-model parameters.

    Parameters
    ----------
    beta : float, optional
        Dust spectral index. Default ``1.6``.
    Td_kelvin : float, optional
        Dust temperature in Kelvin. Default ``19.6``.
    nu0_ghz : float, optional
        Reference frequency in GHz. Default ``353.0`` (Planck 353).
    """

    beta: float = 1.6
    Td_kelvin: float = 19.6
    nu0_ghz: float = 353.0


@dataclass(frozen=True)
class BandConfig:
    """Per-band parameters within an :class:`InstrumentConfig`.

    Parameters
    ----------
    freq : float
        Nominal band center frequency in GHz.
    beam_fwhm_arcmin : float
        Gaussian beam FWHM in arcminutes.
    eff_freq_cmb : float or None, optional
        Bandpass-weighted effective frequency for CMB SED integration, GHz.
    eff_freq_dust : float or None, optional
        Bandpass-weighted effective frequency for dust MBB integration, GHz.
    """

    freq: float
    beam_fwhm_arcmin: float
    eff_freq_cmb: float | None = None
    eff_freq_dust: float | None = None


@dataclass(frozen=True)
class InstrumentConfig:
    """Instrument-wide parameters plus a list of bands.

    Parameters
    ----------
    name : str
        Instrument label (e.g. ``"Planck"``, ``"SO_SAT"``). Reused as the
        ``Tracer.instrument`` field.
    map_template : str
        Path template (``str.format``-style) for per-band FITS maps. Must use
        the named placeholder ``{freq}`` (e.g. ``map_{freq}.fits``,
        ``map_f{freq:03d}.fits``); the tracer's integer frequency is
        interpolated.
    unit_scale : float, optional
        Multiplicative scale applied to raw maps at load time. Default
        ``1e6`` (K -> muK).
    pixelization : str, optional
        ``"healpix_equatorial"`` (default) or ``"car"``. Selects the loader
        in :func:`cmb_diagnostics.io.loaders.get_loader`.
    purify_e : bool, optional
        Forward to ``NmtField(purify_e=...)``.
    purify_b : bool, optional
        Forward to ``NmtField(purify_b=...)``.
    bands : list of BandConfig, optional
        Per-band configuration. Must be non-empty when used in the pipeline.
    """

    name: str
    map_template: str
    unit_scale: float = 1e6
    pixelization: str = "healpix_equatorial"
    purify_e: bool = False
    purify_b: bool = False
    bands: list[BandConfig] = field(default_factory=list)


@dataclass(frozen=True)
class PolAngleConfig:
    """Polarization-angle estimator parameters.

    Parameters
    ----------
    lmin : int, optional
        Lower ell cutoff applied to every lmax sweep window. Default ``30``.
    lmax_sweep : list of int, optional
        Upper ell caps to sweep over; one fit is emitted per cap.
    """

    lmin: int = 30
    lmax_sweep: list[int] = field(default_factory=lambda: [200, 250, 300, 350, 400, 450, 500])


@dataclass(frozen=True)
class AdvancedConfig:
    """Advanced/diagnostic toggles.

    Parameters
    ----------
    fisher_error_form : str, optional
        Reserved for future Fisher-error variants; currently only ``"v1"``
        is honoured.
    write_diagnostic_plots : bool, optional
        When truthy, :meth:`Pipeline.run` emits per-result diagnostic PNGs
        under ``cfg.output_dir/diagnostics/``.
    """

    fisher_error_form: str = "v1"
    write_diagnostic_plots: bool = False


@dataclass(frozen=True)
class Config:
    """Top-level, frozen configuration for a full diagnostics run.

    Construct from a YAML file (:meth:`from_yaml`) or an inline dict
    (:meth:`from_dict`). Every field except ``advanced`` is required by the
    loader.

    Parameters
    ----------
    schema_version : str
        Must equal :data:`SCHEMA_VERSION` (currently ``"1"``).
    output_dir : Path
        Directory where ``.npz`` and ``.png`` artifacts are written.
    nside : int
        HEALPix resolution used throughout the pipeline.
    bandpowers : BandpowersConfig
        Bandpower binning.
    mask : MaskConfig
        Mask construction.
    camb : CambConfig
        CAMB reference loader.
    dust : DustConfig
        MBB dust parameters.
    planck : InstrumentConfig
        Planck instrument and bands.
    so : InstrumentConfig
        Simons Observatory instrument and bands.
    pol_angle : PolAngleConfig
        Polarization-angle sweep parameters.
    advanced : AdvancedConfig, optional
        Optional toggles; defaults to all-off.
    """

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
        """Load a :class:`Config` from a YAML document.

        Parameters
        ----------
        path : str or Path
            Path to a YAML file.

        Returns
        -------
        Config
            Fully validated configuration.

        Raises
        ------
        ConfigError
            If the file is missing, the top-level document is not a mapping,
            or any required field is absent / malformed.
        """
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
        """Build a :class:`Config` from an in-memory mapping.

        Equivalent to :meth:`from_yaml` but skips the file read — useful for
        notebook use and unit tests.

        Parameters
        ----------
        data : dict of str to Any
            Nested config dictionary matching the YAML schema.

        Returns
        -------
        Config
            Fully validated configuration.
        """
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
