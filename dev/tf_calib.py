from .utils import *
import healpy as hp
import pymaster as nmt
import numpy as np
import itertools
import pygsm
import scipy.optimize as opt
import matplotlib.pyplot as plt

class TF_Calib:
    def __init__(self, nside=512):

        self.nside = nside

        # initialize planck
        self.planck_freqs = np.array([100, 143, 217, 353])
        self.planck_eff_freqs = np.array([101.31, 142.71, 221.915, 361.29])
        self.planck_dust_eff_freqs = np.array([105.25, 148.235, 229.097, 372.193])
        self.planck_beams = np.array([9.66, 7.27, 5.01, 4.86])

        self.planck_fname = '/home/yl9946/projects/tp_leakage/planck_equatorial/planck_{}_equatorial_rm_mnp_dp.fits'
        self.planck_f2 = {}
        
        # initialize so
        self.so_fname = "/scratch/gpfs/sa5705/shared/SO_SAT/satp3_maps/cmb_maps_satp3_20240714/map_f{freq:03d}_muKcmb.fits"
        self.so_freqs = np.array([90, 150])
        self.so_beams = np.array([27.4, 17.6])
        self.so_f2 = {}
        
        # mask
        self.mask = None
        self.fsky = None

        # namaster power spectra parameters
        self.bins = None
        self.bin_width = None
        self.e_l = None
        self.e_dl2cl = None
        self.msk = None

        # camb reference values
        self.camb_dl_path = "/scratch/gpfs/yl9946/basic_science/camb_lens_nobb.dat"
        self.camb_dl = None

        # planck EE
        self.planck_ee = {}
        self.planck_xspec_ee = {}

        self.so_auto = {}
        self.so_x_planck = {}
        self.so_x_planck_var = {}

        self.tf = None

        self.result_dir = "/home/yl9946/projects/cmb_diagnostics/result/"

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
        self.camb_dl = dl_bin

    def init_planck_f2(self):
        print("init planck spin 2 fields")
        for f, fwhm in zip(self.planck_freqs, self.planck_beams):
            print(f)
            gbeam = hp.gauss_beam(fwhm / 60 / 180 * np.pi, self.nside * 3 - 1)
            plk_qu = hp.read_map(self.planck_fname.format(f), field=[1, 2]) * 1e6
            plk_qu = hp.ud_grade(plk_qu, 512)
            self.planck_f2[f] = nmt.NmtField(self.mask, plk_qu, beam=gbeam, spin=2)

    def calc_planck_specs(self):
        print("calculating planck spectra")
        for fq1, fq2 in itertools.combinations_with_replacement(self.planck_freqs, 2):
            print(f"p{fq1}xp{fq2}")
            nmt_spec = nmt.compute_full_master(self.planck_f2[fq1], self.planck_f2[fq2], self.bins)
            self.planck_ee[f"p{fq1}xp{fq2}"] = nmt_spec[0]

    def calc_planck_xspec_var(self):
        print("calculating planck cross spectra variance")
        plt.figure(dpi=300)
        for fq1, fq2 in itertools.combinations(self.planck_freqs, 2):
            print(f"p{fq1}xp{fq2}")
            cl13 = self.planck_ee[f"p{fq1}xp{fq1}"]
            cl24 = self.planck_ee[f"p{fq2}xp{fq2}"]
            cl14 = cl23 = self.planck_ee[f"p{fq1}xp{fq2}"]
            self.planck_xspec_ee[f"p{fq1}xp{fq2}"] = knox_covar(cl13, cl24, cl14, cl23, self.fsky, \
                                                             self.e_l, self.bin_width)
            plt.errorbar(self.e_l[self.msk], self.planck_ee[f"p{fq1}xp{fq2}"][self.msk], \
                         self.planck_xspec_ee[f"p{fq1}xp{fq2}"][self.msk]**0.5, \
                 label=f"p{fq1}xp{fq2}", marker='o')
        plt.step(self.e_l[self.msk], self.camb_dl[self.msk], c='k', ls='--', where='mid')
        plt.loglog()
        plt.legend()
        plt.ylabel(r"D$\ell$ ($\mu$K$^2$)")
        plt.xlabel(r"$\ell$")
        plt.savefig(f"{self.result_dir}plk_cross_spec.png")


    def init_so_f2(self):
        print("init SO spin 2 fields")
        for fq, fwhm in zip(self.so_freqs, self.so_beams):
            print(fq)
            gbeam = hp.gauss_beam(fwhm / 60 / 180 * np.pi, self.nside * 3 - 1)
            qu_map = read_carr2healpix(self.so_fname.format(freq=fq))[1:]
            self.so_f2[fq] = nmt.NmtField(self.mask, qu_map, beam=gbeam)

    def calc_so_auto(self):
        print("calculate SO auto spectra")
        for f in self.so_freqs:
            print(f)
            self.so_auto[f] = nmt.compute_full_master(self.so_f2[f], self.so_f2[f], self.bins)[0]

    def calc_so_x_planck(self):
        print("calculating planck x SO spectra")
        for fs, fp in itertools.product(self.so_freqs, self.planck_freqs):
            print(f"s{fs}xp{fp}")
            self.so_x_planck[f"s{fs}xp{fp}"] = nmt.compute_full_master(self.so_f2[fs], self.planck_f2[fp], self.bins)[0]
            plt.loglog(self.e_l[self.msk], self.so_x_planck[f"s{fs}xp{fp}"][self.msk], label=f"s{fs}xp{fp}")


    def calc_so_x_planck_var(self):
        print("calculating SO x planck variance")
        plt.figure(dpi=300)
        plt.step(self.e_l[self.msk], self.camb_dl[self.msk], c='k', ls='--', where='mid')
        for fs, fp in itertools.product(self.so_freqs, self.planck_freqs):
            print(f"s{fs}xp{fp}")
            cl13 = self.so_auto[fs]
            cl24 = self.planck_ee[f"p{fp}xp{fp}"]
            cl14 = cl23 = self.so_x_planck[f"s{fs}xp{fp}"]
            self.so_x_planck_var[f"s{fs}xp{fp}"] = knox_covar(cl13, cl24, cl14, cl23, self.fsky, \
                                                              self.e_l, self.bin_width)
            plt.errorbar(self.e_l[self.msk],  self.so_x_planck[f"s{fs}xp{fp}"][self.msk], \
                         self.so_x_planck_var[f"s{fs}xp{fp}"][self.msk]**0.5, label=f"s{fs}xp{fp}")
        plt.loglog()
        plt.legend()
        plt.savefig(self.result_dir + "so_x_planck_ee.png")

    def estimate_tf(self):
        print("Estimate transfer function")
        tf_res = np.zeros((self.msk.sum(), 7))
        for bin_ind in range(self.msk.sum()):
            # data container created for holding values for fitting
            # f1, f2, ef1, ef2, spec, spec_err
            freq_data_holder_full = np.zeros((14, 6))

            # grab information from planck cross spectra
            for i, ((f1, f2), (fd1, fd2)) in enumerate(zip(itertools.combinations(self.planck_freqs, 2), \
                                                           itertools.combinations(self.planck_dust_eff_freqs, 2))):
                freq_data_holder_full[i] = f1, f2, fd1, fd2, self.planck_ee[f"p{f1}xp{f2}"][self.msk][bin_ind], \
                    self.planck_xspec_ee[f"p{f1}xp{f2}"][self.msk][bin_ind]
            
            # grab info from so x planck spectra            
            for i, (fs, (fp, fdp)) in enumerate(itertools.product(self.so_freqs, zip(self.planck_freqs, self.planck_dust_eff_freqs))):
                freq_data_holder_full[i + 6] = fs, fp, fs, fdp, \
                    self.so_x_planck[f"s{fs}xp{fp}"][self.msk][bin_ind], \
                    self.so_x_planck_var[f"s{fs}xp{fp}"][self.msk][bin_ind]
            dust_unity = dust_dl(1., freq_data_holder_full[:, 2], freq_data_holder_full[:, 3], 1.53) \
                * pygsm.trj2tcmb(freq_data_holder_full[:, 2]) * pygsm.trj2tcmb(freq_data_holder_full[:, 3])
            ee = self.camb_dl[self.msk][bin_ind]
            res = opt.minimize(dust_neg_lnlike, 1, (dust_unity[:6], ee, freq_data_holder_full[:6, 4], \
                                                    freq_data_holder_full[:6, 5]))

            a = res.x[0]
            a_var = hess_inv(dust_unity[:6], freq_data_holder_full[:6, 5])
            # print(res)
            
            res90 = opt.minimize(tf_neg_lnlike, 0.5, (a, dust_unity[6:10], ee, freq_data_holder_full[6:10, 4], \
                                                      freq_data_holder_full[6:10, 5]))
            tf90= res90.x[0]**2
            dmdq = a * dust_unity[6:10] + ee
            tmp_var = hess_inv(dmdq, freq_data_holder_full[6:10, 5])
            tf90_var = 4 * res90.x[0]**2 * tmp_var
            
            res150 = opt.minimize(tf_neg_lnlike, 0.5, (a, dust_unity[10:], ee, freq_data_holder_full[10:, 4], \
                                                       freq_data_holder_full[10:, 5]))
            tf150= res150.x[0]**2
            dmdq = a * dust_unity[10:] + ee
            tmp_var = hess_inv(dmdq, freq_data_holder_full[10:, 5])
            tf150_var = 4 * res150.x[0]**2 * tmp_var
            
            tf_res[bin_ind] = self.e_l[self.msk][bin_ind], a, a_var, tf90, tf90_var, tf150, tf150_var
        self.tf = tf_res.T

    def plot_and_save_result(self):
        plt.figure(dpi=300)
        plt.errorbar(self.tf[0], self.tf[3], self.tf[4]**0.5, ls='--', marker='.', label="90")
        plt.errorbar(self.tf[0] + 2, self.tf[5], self.tf[6]**0.5, ls='--', marker='.', label="150")
        plt.ylabel('transfer func')
        plt.loglog()
        plt.legend()
        plt.savefig(self.result_dir + "transfer_function.png")
        selected_cols = [0, 3, 4, 5, 6]
        np.save(self.result_dir + "tf_ee.npy", self.tf[selected_cols].T)

    def calc_tf_ee(self):
        if self.mask is None:
            raise RuntimeError("Mask needs to be initialized.")
        self.init_nmt()
        self.init_camb_dl()
        self.init_planck_f2()
        self.calc_planck_specs()
        self.calc_planck_xspec_var()
        self.init_so_f2()
        self.calc_so_auto()
        self.calc_so_x_planck()
        self.calc_so_x_planck_var()
        self.estimate_tf()
        self.plot_and_save_result()