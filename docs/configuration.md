# Configuration

A run is fully described by one YAML file. The file is parsed into a typed `Config` dataclass; every field is validated at load time. No code edits, no attribute-monkey-patching.

## Schema

```yaml
schema_version: "1"                 # required; bumped on breaking changes

output_dir: /path/to/outputs        # created if it does not exist

nside: 512                          # target HEALPix nside for everything

bandpowers:
  bin_width: 20                     # linear bin width, in ℓ
  lmin: 30                          # lower ℓ for estimator mask
  lmax: 300                         # upper ℓ for estimator mask
  is_Dell: true                     # nmt.NmtBin.from_nside_linear flag

mask:
  kind: file                        # "file" | "boxes"
  # --- if kind: file ---
  path: /path/to/mask.fits          # CAR or HEALPix (auto-detected by extension/content)
  apodize: true                     # if true, apply the same apodize_square_mask pipeline
  threshold: 1.0e-5                 # zero-out pixels below this weight
  # --- if kind: boxes ---
  # boxes:
  #   - [[-50, 50], [-30, 90]]       # [[dec_min, ra_min], [dec_max, ra_max]] in degrees
  #   - [[-20, -165], [0, -130]]

camb:
  path: /path/to/camb_lens_nobb.dat
  format: bbpower_nobb              # only supported format currently

dust:
  beta: 1.6
  Td_kelvin: 19.6
  nu0_ghz: 353.0

planck:
  name: Planck                      # used as Tracer.instrument
  map_template: /path/planck_{freq}_equatorial_rm_mnp_dp.fits
  unit_scale: 1.0e6                 # K → μK
  pixelization: healpix_equatorial  # "healpix_equatorial" | "car"
  purify_e: true
  purify_b: true
  bands:
    - {freq: 100, beam_fwhm_arcmin: 9.66,  eff_freq_cmb: 101.31,  eff_freq_dust: 105.25 }
    - {freq: 143, beam_fwhm_arcmin: 7.27,  eff_freq_cmb: 142.71,  eff_freq_dust: 148.235}
    - {freq: 217, beam_fwhm_arcmin: 5.01,  eff_freq_cmb: 221.915, eff_freq_dust: 229.097}
    - {freq: 353, beam_fwhm_arcmin: 4.86,  eff_freq_cmb: 361.29,  eff_freq_dust: 372.193}

so:
  name: SO_SAT
  map_template: /path/map_f{freq:03d}_muKcmb.fits
  unit_scale: 1.0e6
  pixelization: car
  purify_e: false
  purify_b: false
  bands:
    - {freq: 90,  beam_fwhm_arcmin: 27.4}
    - {freq: 150, beam_fwhm_arcmin: 17.6}

pol_angle:
  lmin: 30
  lmax_sweep: [200, 250, 300, 350, 400, 450, 500]

# Optional: override internal numerical knobs. Omit unless you know why.
advanced:
  fisher_error_form: v1             # "v1" (hess_inv on full model) | "v2" (rttf_fisher)
  write_diagnostic_plots: false     # per-bin dust/tf fit PNGs
```

## Field-by-field

### Top-level

- **`schema_version`** (required, string). Bumped on incompatible YAML changes. Loader validates.
- **`output_dir`** (required, path). All `.npy`, `.npz`, `.png`, and logs land here. Created recursively. Existing files are overwritten (no numbered suffixes).
- **`nside`** (int, default 512). Target HEALPix resolution. All maps `ud_grade`'d to this.

### `bandpowers`

- `bin_width`: linear Δℓ for NaMaster bin object.
- `lmin`, `lmax`: estimator-side ℓ mask. Note this is *not* the NaMaster band-limit — NaMaster uses `3·nside - 1`. The mask is applied after binning to restrict fit range.
- `is_Dell`: passed to `nmt.NmtBin.from_nside_linear(..., is_Dell=...)`. When `true`, NaMaster returns Dℓ; downstream code converts via `2π/(ℓ(ℓ+1))`.

### `mask`

Two forms. Use `kind: file` for production (analysis masks, survey masks). Use `kind: boxes` to build a rectangular RA/Dec mask on the fly (convenient for clean-patch studies).

- `path`: HEALPix FITS (read via `healpy.read_map`) or CAR FITS (read via `pixell.enmap.read_fits` and reprojected via `reproject.map2healpix`). Auto-detected by first opening the file.
- `apodize`: whether to run the mask through `apodize_square_mask` (healpy smoothing + NaMaster `mask_apodization` C2 at 10°). For analysis masks that are already apodized, set `false`.
- `threshold`: pixels with weight below this are zeroed. Matches the `mask *= mask > 1e-5` idiom in `test/new_container_test.py`.
- `boxes`: list of `[[dec_min, ra_min], [dec_max, ra_max]]` in degrees. Internally converted to radians. Union of boxes is used.

### `camb`

- `path`: BBPower-style text file with rows `[ℓ, TT, EE, BB, TE]` in Dℓ starting at ℓ=2.
- `format`: currently only `bbpower_nobb` supported. Zero-row-prepend and column selection are hard-coded for this format; other formats (e.g. CAMB direct `*.dat`) require a new format handler.

### `dust`

- `beta`, `Td_kelvin`, `nu0_ghz`: MBB shape parameters held fixed during fitting. Only dust *amplitude* is fit per ℓ.

### `planck` / `so` (`InstrumentConfig`)

- `name`: used as `Tracer.instrument`. Must be unique across instruments in one config.
- `map_template`: Python `.format()` string with `{freq}` (or `{freq:03d}`) token. Loader substitutes the band's `freq`.
- `unit_scale`: multiplied into raw map after read. 1e6 for typical Planck/SO maps (K → μK).
- `pixelization`: `healpix_equatorial` (read via `healpy.read_map`) or `car` (read via pixell, reprojected to HEALPix). Other values raise at load time.
- `purify_e`, `purify_b`: passed to `nmt.NmtField` on spin-2 construction. Recommended `true` for Planck, `false` for SO in current practice; the config exposes both.
- `bands`: list of `{freq, beam_fwhm_arcmin, eff_freq_cmb, eff_freq_dust}`. `eff_freq_*` may be omitted if the band is not used in any estimator that requires it (e.g. `eff_freq_cmb` is optional in current estimators; `eff_freq_dust` is required for MBB dust fits at that band).

### `pol_angle`

- `lmin`: lower bound for the fit range.
- `lmax_sweep`: list of upper-bound caps. The estimator repeats the fit at each cap and reports all of them (standard diagnostic for low-ℓ foreground contamination).

### `advanced` (optional)

- `fisher_error_form`: which TF-error formula (see `docs/science_reference.md` §4). Default `v1`.
- `write_diagnostic_plots`: if true, `reports.tf.plot_diagnostics` emits per-bin dust/TF fit PNGs to `{output_dir}/diagnostics/`. Default `false`.

## Two worked examples (also in `configs/`)

Both use SO SATp3, but with different masks:

- **`configs/satp3_south.yaml`** — two apodized box masks over clean-CMB regions (south patch + east patch), reproducing the `test/example.py` V1 setup.
- **`configs/iso_satp3.yaml`** — uses the `analysis_mask_satp3.fits` file mask, reproducing the `test/new_container_test.py` V2 setup.

## Validation behavior at load time

- Missing required keys → `ConfigError` with field path (e.g. `"bandpowers.bin_width: required"`).
- Unknown keys → warning, not error (future-proof for advanced keys).
- Paths referenced in config must exist at load time, unless the `--no-validate-paths` CLI flag is set (useful for testing).
- Instrument `bands[*].freq` must be unique within one instrument.
- `eff_freq_dust` must be set for any band used in a dust-fit estimator (enforced at estimator construction time, not at config load).
