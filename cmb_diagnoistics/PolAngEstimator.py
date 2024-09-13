from .utils import *
from .PSContainer import PSContainer
import itertools
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

class PolAngEB(PSContainer):
    def __init__(self, nside=512, fit_low=30):
        PSContainer.__init__(self, nside=nside)
        nxfreq = int((self.so_freqs.shape[0] + 1) * self.so_freqs.shape[0] / 2)
        self.pol_ang = np.zeros((nxfreq, 7))
        self.pol_ang_var = np.zeros((nxfreq, 7))
        self.row_names = []
        self.fit_low = fit_low

    def estimate_pol_ang(self):
        print("Calculate polarization angles")
        for j in range(7):
            fit_high = 200 + j * 50
            tmp_msk = (self.e_l < fit_high) * (self.e_l >  self.fit_low)
            for i, (f1, f2) in enumerate(itertools.combinations_with_replacement(self.so_freqs, 2)):
                template = (self.so_ee[f"s{f1}xs{f2}"] - self.so_bb[f"s{f1}xs{f2}"])[tmp_msk]
                res = opt.minimize(neg_amp_lnlike, 1, \
                                   (template, self.so_eb[f"s{f1}xs{f2}"][tmp_msk], \
                                    self.so_eb_var[f"s{f1}xs{f2}"][tmp_msk]))
                self.pol_ang[i, j] = res2ang(res)
                self.pol_ang_var[i, j] = hess_inv(template, self.so_eb_var[f"s{f1}xs{f2}"][tmp_msk]) \
                    * (self.pol_ang[i, j] / res.x[0])**2
                
    def plot_and_save_pol_angles(self):
        print("Plot and save polarization angle")
        plt.figure(dpi=300)
        pltx = 200 + 50 * np.arange(7)
        for i, (f1, f2) in enumerate(itertools.combinations_with_replacement(self.so_freqs, 2)):
            plt.errorbar(pltx, self.pol_ang[i] * 180 / np.pi, \
                         self.pol_ang_var[i]**0.5 * 180 / np.pi , label=f"s{f1}xs{f2}")
        plt.xlabel(r"Fitting limit $L_{up}$ ($50 < \ell < L_{up}$)")
        plt.ylabel(r"Polarization Angle (deg)")
        plt.axhline(0, c='k', ls='--')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.result_dir}pol_ang.png")
        np.save(f"{self.result_dir}pol_ang.npy", self.pol_ang)
        np.save(f"{self.result_dir}pol_ang_var.npy", self.pol_ang_var)
        np.save(f"{self.result_dir}row_names.npy", np.array(self.row_names))

    def calc_pol_ang(self):
        if self.mask is None:
            raise RuntimeError("Mask needs to be initialized.")
        self.init_nmt()
        self.init_camb_dl()
        self.init_so_f2()
        self.calc_so_pol_specs()
        self.calc_so_eb_var()
        self.estimate_pol_ang()
        self.plot_and_save_pol_angles()
        

