from .diag_utils import *
import healpy as hp
import pymaster as nmt
import numpy as np
import itertools
import matplotlib.pyplot as plt
import os

class PSContainer:
    # container for handling all power spectra
    def __init__(self, nside=512):

        self.nside = nside

        # initialize planck
        self.planck_freqs = np.array([100, 143, 217, 353])
        self.planck_eff_freqs = np.array([101.31, 142.71, 221.915, 361.29])
        self.planck_dust_eff_freqs = np.array([105.25, 148.235, 229.097, 372.193])
        self.planck_beams = np.array([9.66, 7.27, 5.01, 4.86])

        self.planck_fname = '/home/yl9946/projects/tp_leakage/planck_equatorial/planck_{}_equatorial_rm_mnp_dp.fits'
        self.planck_f0 = {}
        self.planck_f2 = {}
        
        # initialize so
        self.so_fname = "/scratch/gpfs/sa5705/shared/SO_SAT/satp3_maps/cmb_maps_satp3_20240714/map_f{freq:03d}_muKcmb.fits"
        self.so_freqs = np.array([90, 150])
        self.so_beams = np.array([27.4, 17.6])
        self.so_f2 = {}
        self.so_f0 = {}
        
        # mask
        self.mask = None
        self.fsky = None

        # namaster power spectra parameters
        self.bins = None
        self.bin_width = None
        self.lmin = 30
        self.lmax = 300
        self.e_l = None
        self.e_dl2cl = None
        self.msk = None

        # camb reference values
        self.camb_dl_path = "/scratch/gpfs/yl9946/basic_science/camb_lens_nobb.dat"
        self.camb_ee = None
        self.camb_te = None

        # planck EE
        self.planck_ee = {}
        self.planck_tt = {}
        self.planck_xspec_ee_var = {}
        self.planck_xspec_te_var = {}
        self.planck_te = {}
        self.planck_te_var = {}


        self.so_ee = {}
        self.so_bb = {}
        self.so_be = {}
        self.so_eb = {}
        self.so_eb_var = {}
        self.so_x_planck_ee = {}
        self.so_x_planck_ee_var = {}
        self.so_x_planck_te = {}
        self.so_x_planck_te_var = {}

        # self.result_dir = "/home/yl9946/projects/cmb_diagnostics/result/"
        print("Results directory is")
        cur_path = os.path.dirname(os.path.abspath(__file__))
        print(os.path.abspath(f"{cur_path}/../result/"))
        self.result_dir = os.path.abspath(f"{cur_path}/../result/") + '/'

    def init_mask(self, mask):
        print("init mask")
        self.mask = mask
        self.fsky = self.mask.sum() / self.mask.shape[0]

    def init_mask_from_box(self, box):
        print("init mask")
        mask = box2hpmask(self.nside, box)
        self.mask = apodize_square_mask(mask)
        self.fsky = self.mask.sum() / self.mask.shape[0]

    def init_nmt(self, bin_width=20, lmin=30, lmax=300):
        print("init NaMaster")
        self.lmin = lmin
        self.lmax = lmax
        self.bins = nmt.NmtBin.from_nside_linear(self.nside, bin_width, is_Dell=True)
        self.bin_width = bin_width
        self.e_l = self.bins.get_effective_ells()
        self.e_dl2cl = 2 * np.pi / self.e_l / (self.e_l + 1)
        self.msk = (self.e_l < lmax) * (self.e_l > lmin)

    def init_camb_dl(self):
        print("read CAMB dl")
        dl_tmp = np.loadtxt(self.camb_dl_path)
        dl_tmp = np.concatenate((np.zeros((1, 5)), dl_tmp))
        dl_bin = self.bins.bin_cell(dl_tmp[:self.nside * 3, 2]) * self.e_dl2cl
        self.camb_ee = dl_bin
        dl_bin = self.bins.bin_cell(dl_tmp[:self.nside * 3, 4]) * self.e_dl2cl
        self.camb_te = dl_bin


    def init_planck_f2(self):
        print("init planck spin 2 fields")
        for f, fwhm in zip(self.planck_freqs, self.planck_beams):
            print(f"|-{f}")
            gbeam = hp.gauss_beam(fwhm / 60 / 180 * np.pi, self.nside * 3 - 1)
            plk_qu = hp.read_map(self.planck_fname.format(f), field=[1, 2]) * 1e6
            plk_qu = hp.ud_grade(plk_qu, 512)
            self.planck_f2[f] = nmt.NmtField(self.mask, plk_qu, beam=gbeam, spin=2)

    def init_planck_f0(self):
        print("init planck spin 0 fields")
        for f, fwhm in zip(self.planck_freqs, self.planck_beams):
            print(f"|-{f}")
            gbeam = hp.gauss_beam(fwhm / 60 / 180 * np.pi, self.nside * 3 - 1)
            plk_t = hp.read_map(self.planck_fname.format(f), field=0) * 1e6
            plk_t = hp.ud_grade(plk_t, 512)
            self.planck_f0[f] = nmt.NmtField(self.mask, [plk_t], beam=gbeam, spin=0)

    def calc_planck_ee(self):
        print("calculating planck EE spectra")
        for fq1, fq2 in itertools.combinations_with_replacement(self.planck_freqs, 2):
            print(f"|-p{fq1}xp{fq2}")
            nmt_spec = nmt.compute_full_master(self.planck_f2[fq1], self.planck_f2[fq2], self.bins)
            self.planck_ee[f"p{fq1}xp{fq2}"] = nmt_spec[0]

    def calc_planck_ee_auto(self):
        print("Calculating planck EE spectra (auto only)")
        for f in self.planck_freqs:
            print(f"|-p{f}")
            nmt_spec = nmt.compute_full_master(self.planck_f2[f], self.planck_f2[f], self.bins)
            self.planck_ee[f"p{f}xp{f}"] = nmt_spec[0]

    def calc_planck_tt_auto(self):
        print("Calculating planck TT spectra (auto only)")
        for f in self.planck_freqs:
            print(f"|-p{f}")
            nmt_spec = nmt.compute_full_master(self.planck_f0[f], self.planck_f0[f], self.bins)
            self.planck_tt[f"p{f}xp{f}"] = nmt_spec[0]

    def calc_planck_te(self):
        print("calculating planck TE spectra")
        for fq1, fq2 in itertools.product(self.planck_freqs, self.planck_freqs):
            if fq1 == fq2:
                continue
            print(f"|-p{fq1}xp{fq2}")
            nmt_spec = nmt.compute_full_master(self.planck_f0[fq1], self.planck_f2[fq2], self.bins)
            self.planck_te[f"p{fq1}xp{fq2}"] = nmt_spec[0]

    def calc_planck_xspec_ee_var(self):
        print("calculating planck EE cross spectra variance")
        plt.figure(dpi=300)
        for fq1, fq2 in itertools.combinations(self.planck_freqs, 2):
            print(f"|-p{fq1}xp{fq2}")
            cl13 = self.planck_ee[f"p{fq1}xp{fq1}"]
            cl24 = self.planck_ee[f"p{fq2}xp{fq2}"]
            cl14 = cl23 = self.planck_ee[f"p{fq1}xp{fq2}"]
            self.planck_xspec_ee_var[f"p{fq1}xp{fq2}"] = knox_covar(cl13, cl24, cl14, cl23, self.fsky, \
                                                             self.e_l, self.bin_width)
            plt.errorbar(self.e_l[self.msk], self.planck_ee[f"p{fq1}xp{fq2}"][self.msk], \
                         self.planck_xspec_ee_var[f"p{fq1}xp{fq2}"][self.msk]**0.5, \
                         label=f"p{fq1}xp{fq2}", marker='o')
        plt.step(self.e_l[self.msk], self.camb_ee[self.msk], c='k', ls='--', where='mid')
        plt.loglog()
        plt.legend()
        plt.ylabel(r"D$\ell$ ($\mu$K$^2$)")
        plt.xlabel(r"$\ell$")
        plt.savefig(f"{self.result_dir}plk_cross_ee.png")

    def calc_planck_xspec_te_var(self):
        print("calculating planck TE cross spectra variance")
        plt.figure()
        for fq1, fq2 in itertools.product(self.planck_freqs, self.planck_freqs):
            if fq1 == fq2:
                continue
            print(f"|-p{fq1}xp{fq2}")
            cl13 = self.planck_tt[f"p{fq1}xp{fq1}"]
            cl24 = self.planck_ee[f"p{fq2}xp{fq2}"]
            cl14 = cl23 = self.planck_te[f"p{fq1}xp{fq2}"]
            self.planck_xspec_te_var[f"p{fq1}xp{fq2}"] = knox_covar(cl13, cl24, cl14, cl23, self.fsky,\
                                                                self.e_l, self.bin_width)
            plt.errorbar(self.e_l[self.msk], np.abs(self.planck_te[f"p{fq1}xp{fq2}"][self.msk]), \
                         self.planck_xspec_te_var[f"p{fq1}xp{fq2}"][self.msk]**0.5, \
                         label=f"t{fq1}xe{fq2}", marker='o')
        plt.step(self.e_l[self.msk], np.abs(self.camb_te[self.msk]), c='k', ls='--', where='mid')
        plt.loglog()
        plt.legend()
        plt.ylabel(r"D$\ell$ ($\mu$K$^2$)")
        plt.xlabel(r"$\ell$")
        plt.title("Planck x Planck")
        plt.savefig(f'{self.result_dir}planck_cross_te.png')

    def init_so_f2(self):
        print("init SO spin 2 fields")
        for fq, fwhm in zip(self.so_freqs, self.so_beams):
            print(f"|-{fq}")
            gbeam = hp.gauss_beam(fwhm / 60 / 180 * np.pi, self.nside * 3 - 1)
            qu_map = read_carr2healpix(self.so_fname.format(freq=fq))[1:]
            self.so_f2[fq] = nmt.NmtField(self.mask, qu_map, beam=gbeam)

    def init_so_f0(self):
        print("init SO spin 0 fields")
        for fq, fwhm in zip(self.so_freqs, self.so_beams):
            print(f"|-{fq}")
            gbeam = hp.gauss_beam(fwhm / 60 / 180 * np.pi, self.nside * 3 - 1)
            t_map = read_carr2healpix(self.so_fname.format(freq=fq))[0]
            self.so_f0[fq] = nmt.NmtField(self.mask, [t_map], beam=gbeam)

    def calc_so_ee_auto(self):
        print("calculate SO EE auto spectra")
        for f in self.so_freqs:
            print(f"|-{f}")
            self.so_ee[f"s{f}xs{f}"] = nmt.compute_full_master(self.so_f2[f], self.so_f2[f], self.bins)[0]

    def calc_so_pol_specs(self):
        print("calculate SO EE, EB, BE, BB")
        for f1, f2 in itertools.combinations_with_replacement(self.so_freqs, 2):
            print(f"|-s{f1}xs{f2}")
            nmt_spec = nmt.compute_full_master(self.so_f2[f1], self.so_f2[f2], self.bins)
            self.so_ee[f"s{f1}xs{f2}"] = nmt_spec[0]
            self.so_bb[f"s{f1}xs{f2}"] = nmt_spec[3]
            self.so_eb[f"s{f1}xs{f2}"] = nmt_spec[1]
            self.so_be[f"s{f1}xs{f2}"] = nmt_spec[2]

    def calc_so_x_planck_ee(self):
        print("calculating planck x SO EE spectra")
        for fs, fp in itertools.product(self.so_freqs, self.planck_freqs):
            print(f"|-s{fs}xp{fp}")
            self.so_x_planck_ee[f"s{fs}xp{fp}"] = nmt.compute_full_master(self.so_f2[fs], self.planck_f2[fp], self.bins)[0]

    def calc_so_x_planck_te(self):
        print("calculating planck x SO TE spectra")
        for fs, fp in itertools.product(self.so_freqs, self.planck_freqs):
            print(f"|-s{fs}xp{fp}")
            self.so_x_planck_te[f"p{fp}xs{fs}"] = nmt.compute_full_master(self.planck_f0[fp], self.so_f2[fs], self.bins)[0]

    def calc_so_x_planck_ee_var(self):
        print("calculating SO x planck EE variance")
        plt.figure(dpi=300)
        plt.step(self.e_l[self.msk], self.camb_ee[self.msk], c='k', ls='--', where='mid')
        for fs, fp in itertools.product(self.so_freqs, self.planck_freqs):
            print(f"|-s{fs}xp{fp}")
            cl13 = self.so_ee[f"s{fs}xs{fs}"]
            cl24 = self.planck_ee[f"p{fp}xp{fp}"]
            cl14 = cl23 = self.so_x_planck_ee[f"s{fs}xp{fp}"]
            self.so_x_planck_ee_var[f"s{fs}xp{fp}"] = knox_covar(cl13, cl24, cl14, cl23, self.fsky, \
                                                              self.e_l, self.bin_width)
            plt.errorbar(self.e_l[self.msk],  self.so_x_planck_ee[f"s{fs}xp{fp}"][self.msk], \
                         self.so_x_planck_ee_var[f"s{fs}xp{fp}"][self.msk]**0.5, label=f"s{fs}xp{fp}")
        plt.loglog()
        plt.legend()
        plt.savefig(self.result_dir + "so_x_planck_ee.png")

    def calc_so_x_planck_te_var(self):
        print("calculating SO x planck TE variance")
        plt.figure(dpi=300)
        plt.step(self.e_l[self.msk], np.abs(self.camb_te[self.msk]), c='k', ls='--', where='mid')
        for fs, fp in itertools.product(self.so_freqs, self.planck_freqs):
            print(f"|-s{fs}xp{fp}")
            cl13 = self.planck_tt[f"p{fp}xp{fp}"]
            cl24 = self.so_ee[f"s{fs}xs{fs}"]
            cl14 = cl23 = self.so_x_planck_te[f"p{fp}xs{fs}"]
            self.so_x_planck_te_var[f"p{fp}xs{fs}"] = knox_covar(cl13, cl24, cl14, cl23, self.fsky, \
                                                              self.e_l, self.bin_width)
            plt.errorbar(self.e_l[self.msk],  np.abs(self.so_x_planck_te[f"p{fp}xs{fs}"][self.msk]), \
                         self.so_x_planck_te_var[f"p{fp}xs{fs}"][self.msk]**0.5, label=f"p{fp}xs{fs}")
        plt.loglog()
        plt.legend()
        plt.savefig(self.result_dir + "so_x_planck_te.png")

    def calc_so_eb_var(self, eb_lmax=500):
        print("calculate SO eb error bar")
        tmp_msk = (self.e_l > self.lmin) * (self.e_l < eb_lmax)
        plt.figure(dpi=300)
        for f1, f2 in itertools.combinations_with_replacement(self.so_freqs, 2):
            print(f"|-s{f1}xs{f2}")
            cl13 = self.so_ee[f"s{f1}xs{f2}"]
            cl24 = self.so_bb[f"s{f1}xs{f2}"]
            cl23 = self.so_be[f"s{f1}xs{f2}"]
            cl14 = self.so_eb[f"s{f1}xs{f2}"]
            self.so_eb_var[f"s{f1}xs{f2}"] = np.abs(knox_covar(cl13, cl24, cl14, cl23, \
                                                               self.fsky, self.e_l, self.bin_width))
            plt.errorbar(self.e_l[tmp_msk], self.so_eb[f"s{f1}xs{f2}"][tmp_msk], \
                         self.so_eb_var[f"s{f1}xs{f2}"][tmp_msk]**0.5, label=f"s{f1}xs{f2}")
        plt.axhline(ls='--', c='k')
        plt.legend(loc='upper left')
        plt.ylabel(r"D$\ell^{EB}$ [$\mu$K$^2$]")
        plt.xlabel(r"$\ell$")
        plt.savefig(f"{self.result_dir}so_eb_specs.png")