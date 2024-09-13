# CMB Diagnostics

This repo contains code to derive calibration information for SO cmb maps.

## Dependencies
- namaster
- scipy
- numpy
- healpy
- pixell

## Installation
Clone the repo, add the path of the parental directory of the repo to PYTHONPATH

## Functions
At the moment, the repo supports:
- Estimating transfer function from EE SO x Planck cross spectra.
- Estimating transfer function from TE SO x Planck cross spectra (Planck T, SO E).
- Estimating polarization angle from SO EB spectra.

## Example
In `test/example.py`. If you're on Della, you can run the codes directly. Resources are hard coded. You will derive the results for 20240714 SAT-P3 maps. Otherwise, you'll need to prepare stuffs below and modify attributes in each class accordingly. See next paragraph for what you need to prepare.

If cmb_util(https://github.com/liuyiqiandrew/cmb_utils) is available on git. Feel free to grab functions directly from repo or add it to your PYTHONPATH.

## What's about the code
Apologies for the horrible user interface, I may consider refining it if I have more spare time. I would strongly suggest skim through `PSContainer` to get a sense in how the code works. In particular, understanding the constructor and initalizer for namaster field `init_planck_f2(self)` will give you a good sense on how to format file name and frequency information.

Currently, all estimator class inherits from `PSContainer` which handles all power spectra information. Estimator is only responsible for the final estimator specific computation (i.e. transfer function, polarization angle etc.) and saving the result.

### What you need to prepare
- Planck maps in equatorial coordinates and HealPix Pixellization (if you're on della, they are availale at `/home/yl9946/projects/tp_leakage/planck_equatorial/planck_*_equatorial_rm_mnp_dp.fits`). You need to modify the path to these maps by, for example
  ```
  tf_ee_est = cmb_diagnostics.TransferFuncEE()
  tf_ee_est.planck_fname = '/home/yl9946/projects/tp_leakage/planck_equatorial/planck_{}_equatorial_rm_mnp_dp.fits'
  ```
  The file name is generic, \{\} is where you distinguish frequencies (but you should specify frequncy in `tf_ee_est.planck_freqs`, add string formating 
  info in \{\} as it fits.). You will also need the effective planck frequency for dust and put them in `tf_ee_est.planck_eff_freqs`
- SO maps, in **carr** pixelization. Support the file name as follows
  ```
  tf_ee_est = cmb_diagnostics.TransferFuncEE()
  tf_ee_est.so_fname = '/scratch/gpfs/sa5705/shared/SO_SAT/satp3_maps/cmb_maps_satp3_20240714/map_f{freq:03d}_muKcmb.fits'
  ```
- Beams for SO and planck. Assume gaussian beams, the values are stored as numpy array in `tf_ee_est.so_beams` and `tf_ee_est.planck_beams`.
- Camb model, something like the 'nobb' template from BBPower(https://github.com/simonsobs/BBPower/blob/main/examples/data/camb_lens_nobb.dat). Supply the path at `tf_ee_est.camb_dl_path`.

## Inquiries
