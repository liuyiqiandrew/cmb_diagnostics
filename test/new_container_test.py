import numpy as np
import healpy as hp
import cmb_diagnostics


def main():
    planck_beams_fwhm = np.array([9.66, 7.27, 5.01, 4.86])
    planck_beams = [hp.gauss_beam(fwhm, 512 * 3) for fwhm in planck_beams_fwhm]
    planck_freqs = np.array([100, 143, 217, 353])
    planck_fnames = [f'/home/yl9946/projects/tp_leakage/planck_equatorial/planck_{f}_equatorial_rm_mnp_dp.fits' for f in planck_freqs]
    print(planck_fnames)
    


if __name__ == "__main__":
    main()