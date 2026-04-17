# Data model

Canonical objects that flow through the pipeline. Each is a simple, immutable-where-possible dataclass. The old codebase used string keys (`"p100xp143"`, `"s90xp150"`) and dict-of-dict indexing; the new model replaces all of those with typed identifiers.

## Identifier objects

### `Tracer`

A frequency-band view of one instrument.

```python
@dataclass(frozen=True)
class Tracer:
    instrument: str      # "Planck", "SO_SAT", ...
    freq: float          # GHz, nominal
    spin: int            # 0 (temperature) or 2 (polarization)

Tracer("Planck", 143.0, spin=2)   # Planck 143 GHz Q/U
Tracer("SO_SAT", 90.0, spin=0)    # SO 90 GHz T
```

Hashable, usable as dict key, replaces string concat like `f"p{fq}"` everywhere.

### `BandInfo`

Physics of one tracer: beam, effective frequencies, unit scale.

```python
@dataclass(frozen=True)
class BandInfo:
    tracer: Tracer
    beam_fwhm_arcmin: float
    eff_freq_cmb: float | None       # spectrum-weighted, CMB SED
    eff_freq_dust: float | None      # spectrum-weighted, dust SED
    unit_scale: float = 1.0          # multiply raw map by this (e.g. 1e6 for K→μK)
```

Current code has four separate lists (`planck_freqs`, `planck_eff_freqs`, `planck_dust_eff_freqs`, `planck_beams`) indexed positionally. `BandInfo` groups them per band.

### `SpectrumKey`

Identifies a single computed Cℓ array.

```python
Comp = Literal["EE", "EB", "BE", "BB", "TE", "TB", "TT"]

@dataclass(frozen=True)
class SpectrumKey:
    tracer1: Tracer
    tracer2: Tracer
    comp: Comp
```

The order convention is: whichever tracer corresponds to the first field slot in NaMaster's `compute_full_master(f_a, f_b)` for this component. The old code had asymmetric conventions (`pAxpB` with A<B for EE, `sFsxpFp` for SO×Planck EE but `pFpxsFs` for TE). This is now captured explicitly by `tracer1` / `tracer2` ordering, with helpers that canonicalize.

## Data containers

### `Mask`

```python
@dataclass(frozen=True)
class Mask:
    hp_map: np.ndarray          # HEALPix, apodized
    nside: int
    fsky_effective: float       # Σ w² / Npix, NOT Σ w / Npix
```

The current code uses `mask.sum() / Npix` as fsky, which over-estimates for apodized masks. The refactor uses the effective fsky `Σw²/Npix` consistently; Knox variance expects this.

### `FieldSet`

All NaMaster fields for one "instrument view" of the sky.

```python
class FieldSet:
    name: str
    fields: dict[Tracer, nmt.NmtField]      # keyed on spin=0 or spin=2 tracer

    def add(self, tracer: Tracer, field: nmt.NmtField) -> None: ...
    def get(self, tracer: Tracer) -> nmt.NmtField: ...
    def tracers(self, spin: int | None = None) -> Iterable[Tracer]: ...
```

Replaces V1's twin `planck_f0` / `planck_f2` / `so_f0` / `so_f2` dicts on `PSContainer`, and V2's `NmtFieldContainer` (which this class is close to, but is typed on `Tracer` instead of bare frequency).

### `Bandpowers`

```python
@dataclass(frozen=True)
class Bandpowers:
    nmt_bin: nmt.NmtBin
    bin_width: int
    lmin: int
    lmax: int
    is_dell: bool

    @property
    def effective_ell(self) -> np.ndarray: ...
    @property
    def msk(self) -> np.ndarray:          # boolean over bins
        return (self.effective_ell > self.lmin) & (self.effective_ell < self.lmax)
    @property
    def dl2cl(self) -> np.ndarray:
        return 2 * np.pi / self.effective_ell / (self.effective_ell + 1)
```

Replaces the scattered `bins`, `e_l`, `bin_width`, `msk`, `e_dl2cl` attributes on `PSContainer`.

### `Spectra`

Power spectra + per-bin variance, indexed by `SpectrumKey`.

```python
class Spectra:
    bandpowers: Bandpowers
    cls: dict[SpectrumKey, np.ndarray]
    vars: dict[SpectrumKey, np.ndarray]

    def add(self, key: SpectrumKey, cl: np.ndarray, var: np.ndarray) -> None: ...
    def get(self, t1: Tracer, t2: Tracer, comp: Comp) -> tuple[np.ndarray, np.ndarray]: ...
    def has(self, t1, t2, comp) -> bool: ...
    def iter_pairs(self, comp: Comp) -> Iterable[tuple[Tracer, Tracer]]: ...
    def save_npz(self, path: Path) -> None: ...
    @classmethod
    def load_npz(cls, path: Path) -> "Spectra": ...
```

Fuses V1's many parallel dicts (`planck_ee`, `planck_xspec_ee_var`, `so_x_planck_ee`, ...) and V2's `Container.PSContainer` (which was typed by `PSType` enum) into a single container that holds *all* component spectra regardless of pstype. Accessors filter by component on the fly.

## Models

### `CMBReference`

```python
class CMBReference:
    bandpowers: Bandpowers
    cls_binned: dict[Comp, np.ndarray]     # 'EE', 'BB', 'TE', 'TT' → Cℓ at effective ℓ

    def get(self, comp: Comp) -> np.ndarray: ...
```

Replaces `self.camb_ee`, `self.camb_te`, etc. Created via `io.camb.load_camb_reference(path, bandpowers)`.

### `DustModel` (protocol)

```python
class DustModel(Protocol):
    def predict_cross(self, t1: Tracer, t2: Tracer, band_info: BandInfoMap) -> np.ndarray:
        """Return dust Cℓ for (t1, t2) at unit amplitude, in CMB μK², at bandpowers."""

    def fit_amplitude(self, spectra: Spectra, cmb_ref: CMBReference,
                      tracer_pairs: Iterable[tuple[Tracer, Tracer]],
                      comp: Comp, ell_idx: int) -> FitAmplitude:
        """Fit dust amplitude at one bandpower over given tracer pairs."""
```

Initial implementation: `MBBDustModel(beta=1.6, Td=19.6, nu0=353.0)` — exactly the current code's shape, with amplitude the only fit parameter.

### `TFAmplitudeModel` / `RotationModel`

Trivial wrappers around the scalar-scaling and (EE−BB)-template models already in `Models.py` / `diag_utils.py`. Exposed for composition.

## Estimation

### `FitResult`

```python
@dataclass
class FitResult:
    name: str                              # "TF_EE_SO_90", etc.
    ell: np.ndarray                        # effective ℓ at evaluated bins
    values: np.ndarray                     # per-bin point estimates
    errors: np.ndarray                     # per-bin 1σ
    diagnostics: dict[str, np.ndarray]     # residuals, χ², auxiliary fits (e.g. dust amp per ℓ)
    metadata: dict[str, Any]               # tracer, config snapshot, timestamp

    def save_npz(self, path: Path) -> None: ...
```

Replaces the loose `self.tf`, `self.dtf`, `self.pol_ang`, `self.pol_ang_var` attributes.

## Concrete example

```python
from cmb_diagnostics import (
    Config, Tracer, Bandpowers, FieldSet, Spectra, SpectrumKey,
    CMBReference, MBBDustModel, TransferFunctionEE,
)

cfg = Config.from_yaml("configs/satp3_south.yaml")

t_p143 = Tracer("Planck", 143.0, spin=2)
t_so90 = Tracer("SO_SAT", 90.0, spin=2)

# ... after fields & spectra are computed:
cl, var = spec_ps.get(t_p143, t_so90, comp="EE")

dust = MBBDustModel(beta=1.6, Td=19.6, nu0=353.0, band_info=cfg.band_info)
tf_ee = TransferFunctionEE(spec_pp, spec_ps, cmb_ref, dust)
result = tf_ee.estimate(target=t_so90)
assert isinstance(result, FitResult)
result.save_npz(cfg.output_dir / "tf_ee_so90.npz")
```
