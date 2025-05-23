import numpy as np
import healpy as hp
import cmb_diagnostics
import pymaster as nmt
from cmb_diagnostics.cmb_diagnoistics.Constants import PSType
from cmb_diagnostics.cmb_diagnoistics.diag_utils import read_carr2healpix
import cmb_utils
import itertools
import matplotlib.pyplot as plt


def main():
    box1 = np.array(((-50,50),(-30,90))) * np.pi / 180 # clean CMB region south
    box2 = np.array(((-20,-165),(0,-130))) * np.pi / 180 # clean CMB region east
    mask1 = cmb_utils.maptools.carrbox2hpmask(512, box1)
    ap_m1 = cmb_utils.maptools.apodize_square_mask(mask1)
    mask2 = cmb_utils.maptools.carrbox2hpmask(512, box2)
    ap_m2 = cmb_utils.maptools.apodize_square_mask(mask2)
    mask = ap_m1 + ap_m2
    fsky = mask.sum() / hp.nside2npix(512)


    planck_beams_fwhm = np.array([9.66, 7.27, 5.01, 4.86])
    planck_beams = [hp.gauss_beam(fwhm / 60 / 180 * np.pi, 512 * 3) for fwhm in planck_beams_fwhm]
    planck_freqs = np.array([100, 143, 217, 353])
    planck_fnames = [f'/home/yl9946/projects/tp_leakage/planck_equatorial/planck_{f}_equatorial_rm_mnp_dp.fits' for f in planck_freqs]
    
    plk_field_container = cmb_diagnostics.Container.NmtFieldContainer("Planck")
    print("Create Planck Fields")
    for idx, fname in enumerate(planck_fnames):
        pmaps_raw = hp.read_map(fname, field=[0, 1, 2]) * 1e6
        pmaps = hp.ud_grade(pmaps_raw, 512)
        f0 = nmt.NmtField(mask, pmaps[:1], beam=planck_beams[idx])
        f2 = nmt.NmtField(mask, pmaps[1:], beam=planck_beams[idx], purify_b=True, purify_e=True)
        plk_field_container.add_spin0_field(planck_freqs[idx], f0)
        plk_field_container.add_spin2_field(planck_freqs[idx], f2)

    so_beam_fwhm = np.array([27.4, 17.6])
    so_beams = [hp.gauss_beam((fwhm / 60 / 180) * np.pi, 512 * 3) for fwhm in so_beam_fwhm]
    so_freqs = np.array([90, 150])
    so_fnames = [f"/scratch/gpfs/sa5705/shared/SO_SAT/satp3_maps/cmb_maps_satp3_20240714/map_f{freq:03d}_muKcmb.fits" for freq in so_freqs]

    so_field_container = cmb_diagnostics.Container.NmtFieldContainer("SO")
    print("Create SO Field")
    for idx, fname in enumerate(so_fnames):
        so_map = read_carr2healpix(fname)
        f0 = nmt.NmtField(mask, so_map[:1], beam=so_beams[idx])
        f2 = nmt.NmtField(mask, so_map[1:], beam=so_beams[idx])
        so_field_container.add_spin0_field(so_freqs[idx], f0)
        so_field_container.add_spin2_field(so_freqs[idx], f2)

    bins = nmt.NmtBin.from_nside_linear(512, 20, True)

    print("Calculate planck auto power spectrum")
    plk_pp_calculator = cmb_diagnostics.PSCalculator(
        plk_field_container, plk_field_container, bins, fsky, [PSType.PP]
    )
    plk_pp_calculator.calculate_spectra()

    print("Calculate so x planck auto power spectrum")
    sp_pp_calculator = cmb_diagnostics.PSCalculator(
        plk_field_container, so_field_container, bins, fsky, [PSType.PP]
    )
    sp_pp_calculator.calculate_spectra()

    plk_pp = plk_pp_calculator.pp_container
    sp_pp = sp_pp_calculator.pp_container

    e_l = plk_pp_calculator.eff_ell
    msk = (e_l > 30) * (e_l < 300)

    print("Plot figure")
    plt.figure()
    for f1, f2 in itertools.combinations(planck_freqs, 2):
        dl = plk_pp.get_spectrum('EE', f1, f2)
        plt.loglog(e_l[msk], dl[msk], label=f"{f1}x{f2}")
    plt.legend()
    plt.savefig('test_planck_auto.png')
    
    plt.figure()
    for f1, f2 in itertools.product(so_freqs, planck_freqs):
        dl = sp_pp.get_spectrum('EE', f1, f2)
        plt.loglog(e_l[msk], dl[msk], label=f"{f1}x{f2}")
    plt.legend()
    plt.savefig("test_sxp.png")



if __name__ == "__main__":
    main()