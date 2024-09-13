from .utils import *
from .PSContainer import PSContainer
import numpy as np
import itertools
import pygsm
import scipy.optimize as opt
import matplotlib.pyplot as plt


class TransferFuncEE(PSContainer):
    def __init__(self, nside=512):
        PSContainer.__init__(self, nside=nside)
        self.tf = np.empty(7)

    def __generate_freq_data(self, bin_ind):
        """ helper for generating information to estimate transfer function """
        holder_len = int(self.planck_freqs.shape[0] * (self.planck_freqs.shape[0] - 1) / 2 \
                         + self.planck_freqs.shape[0] * self.so_freqs.shape[0])
        freq_data_holder_full = np.zeros((holder_len, 6))
        for i, ((f1, f2), (fd1, fd2)) in enumerate(zip(itertools.combinations(self.planck_freqs, 2), \
                                                           itertools.combinations(self.planck_dust_eff_freqs, 2))):
                freq_data_holder_full[i] = f1, f2, fd1, fd2, self.planck_ee[f"p{f1}xp{f2}"][self.msk][bin_ind], \
                    self.planck_xspec_ee_var[f"p{f1}xp{f2}"][self.msk][bin_ind]
            
            # grab info from so x planck spectra            
        for i, (fs, (fp, fdp)) in enumerate(itertools.product(self.so_freqs, \
                                                                zip(self.planck_freqs, \
                                                                    self.planck_dust_eff_freqs))):
            freq_data_holder_full[i + 6] = fs, fp, fs, fdp, \
                self.so_x_planck_ee[f"s{fs}xp{fp}"][self.msk][bin_ind], \
                self.so_x_planck_ee_var[f"s{fs}xp{fp}"][self.msk][bin_ind]
        return freq_data_holder_full

    def estimate_tf(self):
        print("Estimate transfer function")
        tf_res = np.zeros((self.msk.sum(), 7))
        for bin_ind in range(self.msk.sum()):
            # data container created for holding values for fitting
            # f1, f2, ef1, ef2, spec, spec_err
            freq_data_holder_full = self.__generate_freq_data(bin_ind=bin_ind)

            dust_unity = dust_dl(1., freq_data_holder_full[:, 2], freq_data_holder_full[:, 3], 1.53) \
                * pygsm.trj2tcmb(freq_data_holder_full[:, 2]) * pygsm.trj2tcmb(freq_data_holder_full[:, 3])
            ee = self.camb_ee[self.msk][bin_ind]
            res = opt.minimize(dust_neg_lnlike, 1, (dust_unity[:6], ee, freq_data_holder_full[:6, 4], \
                                                    freq_data_holder_full[:6, 5]))

            a = res.x[0]
            a_var = hess_inv(dust_unity[:6], freq_data_holder_full[:6, 5])
            
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

    def plot_and_save_transfer_func(self):
        print("Plot and save transfer function")
        plt.figure(dpi=300)
        plt.errorbar(self.tf[0], self.tf[3], self.tf[4]**0.5, ls='--', marker='.', label="90")
        plt.errorbar(self.tf[0] + 2, self.tf[5], self.tf[6]**0.5, ls='--', marker='.', label="150")
        plt.ylabel('transfer func')
        plt.loglog()
        plt.legend()
        plt.savefig(self.result_dir + "transfer_function_ee.png")
        selected_cols = [0, 3, 4, 5, 6]
        np.save(self.result_dir + "tf_ee.npy", self.tf[selected_cols].T)

    def calc_tf_ee(self):
        if self.mask is None:
            raise RuntimeError("Mask needs to be initialized.")
        self.init_nmt()
        self.init_camb_dl()
        self.init_planck_f2()
        self.calc_planck_ee()
        self.calc_planck_xspec_ee_var()
        self.init_so_f2()
        self.calc_so_ee_auto()
        self.calc_so_x_planck_ee()
        self.calc_so_x_planck_ee_var()
        self.estimate_tf()
        self.plot_and_save_transfer_func()


class TransferFuncTE(PSContainer):
    def __init__(self, nside=512):
        PSContainer.__init__(self, nside=nside)
        self.tf = np.empty(7)
    
    def __generate_freq_info(self, bin_ind):
        """ helper for generating information to estimate transfer function """
        holder_len = int(self.planck_freqs.shape[0] * (self.planck_freqs.shape[0] - 1) \
                         + self.planck_freqs.shape[0] * self.so_freqs.shape[0])
        fq_data_holder_full = np.zeros((holder_len, 6))
        enum_adjust = 0
        for i, ((fq1, fq2), (fd1, fd2)) in enumerate(zip(itertools.product(self.planck_freqs, self.planck_freqs), \
                                                            itertools.product(self.planck_dust_eff_freqs, \
                                                                            self.planck_dust_eff_freqs))):
            if fq1 == fq2:
                enum_adjust += 1
                continue
            fq_data_holder_full[i - enum_adjust] = fq1, fq2, fd1, fd2, self.planck_te[f"p{fq1}xp{fq2}"][self.msk][bin_ind], \
                self.planck_xspec_te_var[f"p{fq1}xp{fq2}"][self.msk][bin_ind]
        
        for i, (fs, (fp, fdp)) in enumerate(itertools.product(self.so_freqs, zip(self.planck_freqs, self.planck_dust_eff_freqs))):
            fq_data_holder_full[i + 12] = fs, fp, fs, fdp, \
                self.so_x_planck_te[f"p{fp}xs{fs}"][self.msk][bin_ind], \
                self.so_x_planck_te_var[f"p{fp}xs{fs}"][self.msk][bin_ind]
        return fq_data_holder_full

    def estimate_tf(self):
        tf_res = np.zeros((self.msk.sum(), 7))
        for bin_ind in range(self.msk.sum()):
            fq_data_holder_full = self.__generate_freq_info(bin_ind=bin_ind)

            dust_unity = dust_dl(1., fq_data_holder_full[:, 2], fq_data_holder_full[:, 3], 1.53) \
            * pygsm.trj2tcmb(fq_data_holder_full[:, 2]) * pygsm.trj2tcmb(fq_data_holder_full[:, 3]) # in CMB unit
            
            te = self.camb_te[self.msk][bin_ind]
            
            res = opt.minimize(dust_neg_lnlike, 1, (dust_unity[:12], te, fq_data_holder_full[:12, 4], \
                                                    fq_data_holder_full[:12, 5]))
            a = res.x[0]
            a_var = hess_inv(a * dust_unity[:12] + te, fq_data_holder_full[:12, 5])
            # print(res)
            
            res90 = opt.minimize(tf_neg_lnlike, 0.5, (a, dust_unity[12:16], te, fq_data_holder_full[12:16, 4], \
                                                    fq_data_holder_full[12:16, 5]))
            tf90= res90.x[0]**2
            dmdq = a * dust_unity[12:16] + te
            tmp_var = hess_inv(dmdq, fq_data_holder_full[12:16, 5])
            tf90_var = 4 * res90.x[0]**2 * tmp_var
            
            res150 = opt.minimize(tf_neg_lnlike, 0.5, (a, dust_unity[16:], te, fq_data_holder_full[16:, 4], \
                                                    fq_data_holder_full[16:, 5]))
            tf150= res150.x[0]**2
            dmdq = a * dust_unity[16:] + te
            tmp_var = hess_inv(dmdq, fq_data_holder_full[16:, 5])
            tf150_var = 4 * res150.x[0]**2 * tmp_var
            
            tf_res[bin_ind] = self.e_l[self.msk][bin_ind], a, a_var, tf90, tf90_var, tf150, tf150_var

        self.tf = tf_res.T

    def plot_and_save_transfer_func(self):
        print("Plot and save transfer function")
        plt.figure(dpi=300)
        plt.errorbar(self.tf[0], self.tf[3], self.tf[4]**0.5, ls='--', marker='.', label="90")
        plt.errorbar(self.tf[0] + 2, self.tf[5], self.tf[6]**0.5, ls='--', marker='.', label="150")
        plt.ylabel('transfer func')
        plt.loglog()
        plt.legend()
        plt.savefig(self.result_dir + "transfer_function_te.png")
        selected_cols = [0, 3, 4, 5, 6]
        np.save(self.result_dir + "tf_te.npy", self.tf[selected_cols].T)

    def calc_tf_te(self):
        if self.mask is None:
            raise RuntimeError("Mask needs to be initialized.")
        self.init_nmt(lmin=50)
        self.init_camb_dl()
        self.init_planck_f2()
        self.init_planck_f0()
        self.calc_planck_ee_auto()
        self.calc_planck_te()
        self.calc_planck_tt_auto()
        self.calc_planck_xspec_te_var()
        self.init_so_f2()
        self.init_so_f0()
        self.calc_so_ee_auto()
        self.calc_so_x_planck_te()
        self.calc_so_x_planck_te_var()
        self.estimate_tf()
        self.plot_and_save_transfer_func()