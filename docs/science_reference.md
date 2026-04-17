# Science reference

Formulas the refactor preserves verbatim from the existing implementation. Each entry points at where the current code encodes it, so regression tests can pin behavior before the refactor begins.

## 1. Map-level model for a filtered experiment

Let the true sky be a scalar field `T(n̂)`. Each instrument observes:

- **Planck:**  `T_P = B_P * T  + noise_P`     (deconvolved via NaMaster `beam=` argument)
- **SO:**      `T_S = F · (B_S * T) + noise_S`, where `F` is the map-maker transfer function at map level.

Define the power-level transfer function:

```
TF_Cℓ = F²
```

The observed SO × Planck cross-spectrum is then:

```
⟨C_ℓ(SO × Planck)⟩ = F · C_ℓ(true × true) = √TF_Cℓ · C_ℓ(true)
```

So √TF is the amplitude fit against SO × Planck; squaring gives the power-level TF.

Source in current code: `SOPlkTF.__tf_ee` lines 99–153 (`tf[i] = tf_fitter.fit_result.x[0]**2`) and `TransferFuncEE.estimate_tf` (`tf90 = res90.x[0]**2`).

## 2. Dust foreground model

Modified black-body (MBB) amplitude model at cross-frequencies (ν₁, ν₂):

```
D_ℓ^dust(ν₁, ν₂) = a_d · (ν₁ ν₂ / ν₀²)^β_d · (B(T_d, ν₁) B(T_d, ν₂)) / B(T_d, ν₀)² · g(ν₁) g(ν₂)
```

where:
- `a_d` — fit amplitude per ℓ-bin
- `β_d = 1.6` (V2) or `1.53` (V1)
- `T_d = 19.6` K
- `ν₀ = 353` GHz
- `B(T, ν)` — Planck function at temperature T, frequency ν
- `g(ν) = trj2tcmb(ν)` — Rayleigh–Jeans to thermodynamic-CMB conversion

Sources: `Models.amp_dust_mbb`, `diag_utils.dust_dl`.

The refactor sets `β_d = 1.6` by default (V2's value), but it's configurable via `cfg.dust.beta`.

## 3. CMB reference

Input: BBPower-style `camb_lens_nobb.dat`, columns `[ℓ, TT, EE, BB, TE]` as Dℓ, starting at ℓ = 2.

Pre-processing (identical in both current generations):
1. Prepend one row of zeros to shift indexing to start at ℓ = 0.
2. Slice `[:3·nside]` to bound at the band limit.
3. Call `nmt.NmtBin.bin_cell(dl_arr)` to bin-average.
4. Multiply by `e_dl2cl = 2π / ℓ_eff / (ℓ_eff + 1)` to convert from Dℓ to Cℓ in the binned space.

Sources: `PSContainer.init_camb_dl`, `SOPlkTF.__init__`.

## 4. Transfer function — EE estimator

For each bin index `i`:

1. **Dust amplitude fit** on Planck × Planck residuals, using only the 6 Planck frequency pairs:

   `argmin_{a} Σ_{pairs} (⟨EE⟩_{p1,p2}(ℓ_i) − C^CMB_EE(ℓ_i) − a · dust_unit_{p1,p2})² / var`

   where `dust_unit_{p1,p2}` is the MBB shape at unit amplitude. V2 filters out bins where the Planck residual is non-positive before the fit.

2. **TF amplitude fit** on SO × Planck using the dust model evaluated at SO-side frequencies:

   `argmin_{r} Σ_{p} (⟨EE⟩_{p,SO}(ℓ_i) − r · (C^CMB_EE(ℓ_i) + a_dust · dust_unit_{p,SO}))² / var`

3. Report `TF(ℓ_i) = r²`.

4. Error propagation: Fisher information on `r`, chain-ruled to TF:
   - V1: `tf_var = 4 · r² · hess_inv(model, var)` where `hess_inv(dmdq, var) = 1/Σ(dmdq²/var)` and `dmdq = a·dust_unit + C^CMB_EE`.
   - V2: `dtf = 2·√TF · rttf_error`, where `rttf_error = 1/√Σ(pxp_est²/dpxs²)`.

The two forms differ: V1 uses the *measured* SO × Planck uncertainty scaled by (model prediction)² summed across frequency pairs; V2 uses (model prediction)² directly (not (model)²·var, but model²/var). In practice they are close on well-measured bins. The refactor adopts V1's form by default (matches `hess_inv` derivation from the χ² Hessian cleanly) and exposes V2's as an option for comparison.

Sources: `TransferFuncEstimator.py` (V1), `Estimator.SOPlkTF` (V2), `diag_utils.dust_neg_lnlike`, `diag_utils.tf_neg_lnlike`, `diag_utils.hess_inv`, `Models.rttf_error`.

## 5. Transfer function — TE estimator

Structurally identical to EE, with:
- Dust fit over 12 Planck TE pairs (ordered pairs excluding auto).
- CMB reference: TE instead of EE.
- SO side uses E mode, Planck side uses T.
- `lmin = 50` (not 30), motivated by low-ℓ TE cosmic variance.

Source: `TransferFuncEstimator.TransferFuncTE`.

V2's `SOPlkTF.__tf_te` is currently a stub; the refactor's `TransferFunctionTE` ports V1's logic into the new architecture.

## 6. Polarization angle estimator

For a rotation by angle α of the SO polarization reference axis, the induced EB spectrum is:

```
EB_obs(ℓ) = ½ · (EE_true(ℓ) − BB_true(ℓ)) · sin(4α)
```

In the small-angle regime, `sin(4α) ≈ 4α`, so fitting

```
EB_obs(ℓ) = a · (EE_obs(ℓ) − BB_obs(ℓ))
```

gives `α = arctan(2a) / 4`.

(Deriving: sin(4α)/2 is the coefficient; equating sin(4α)/2 = a gives α = arcsin(2a)/4; for small a this approximates to arctan(2a)/4, which is what the code uses. Formally `res2ang(res) = arctan(a·2)/4`.)

Error propagation: Fisher on `a`, scaled by `(α / a)²` to approximate the transformation jacobian near the fit.

The pipeline sweeps the upper fit cap in {200, 250, 300, 350, 400, 450, 500} to check stability.

Sources: `PolAngEstimator.PolAngEB`, `diag_utils.res2ang`, `diag_utils.neg_amp_lnlike`.

## 7. Knox covariance

Bandpower variance approximation (assuming Gaussian fields, disconnected trispectrum):

```
Var(Ĉ_ℓ^{AB}) = (C_ℓ^{AA} · C_ℓ^{BB} + (C_ℓ^{AB})²) / ((2ℓ + 1) · f_sky · Δℓ)
```

For cross-pairs between two different cross-spectra (as used in dust / TF likelihoods), the symmetric form is:

```
Var = (C_ℓ^{13} · C_ℓ^{24} + C_ℓ^{14} · C_ℓ^{23}) / ((2ℓ + 1) · f_sky · Δℓ)
```

Source: `diag_utils.knox_covar`.

The refactor replaces `f_sky = Σw/Npix` (raw mask sum / pixel count) with the effective apodized value `f_sky_effective = Σw²/Npix`. This is a small correction for apodized masks and is standard practice.

Future extension point: swap Knox for the full NaMaster Gaussian covariance (`nmt.NmtCovarianceWorkspace`) behind the same `spectra/covariance.py` interface.

## 8. Pixelization / coordinate conventions

- **Planck** maps: equatorial, HEALPix, nside arbitrary on disk, `ud_grade` to target nside.
- **SO** maps: CAR (from pixell), converted to HEALPix via `pixell.reproject.map2healpix(method='spline', order=1)`, then `ud_grade`.
- All intermediate computation at a configurable `nside` (default 512).
- Units: raw maps in K, multiplied by `unit_scale` (default 1e6) to produce μK before field construction. All Cℓ in μK².

Source: `diag_utils.read_carr2healpix`, `PSContainer.init_planck_f2`, `PSContainer.init_so_f2`.

## 9. Numerical conventions

- Beams: Gaussian-only, `hp.gauss_beam(fwhm_rad, 3·nside - 1)`.
- Bandpower binning: `nmt.NmtBin.from_nside_linear(nside, bin_width, is_Dell=True)`. The `is_Dell=True` flag means NaMaster returns bin-averaged Dℓ; the code multiplies by `e_dl2cl` to convert to Cℓ.
- `purify_e` / `purify_b` flags on `nmt.NmtField` default to True for Planck spin-2 and False for SO in current tests (`test/new_container_test.py:41–42` vs `:58`). The refactor makes this per-instrument configurable.
