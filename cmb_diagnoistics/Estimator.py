import scipy.optimize as opt
import numpy as np
from .Container import PSContainer
from .Constants import PSType
import pymaster as nmt

class Fitter:
    def __init__(self, func, p0, x_vals, y_vals, args=None, dy=None):
        self.func = func
        self.p0 = p0
        self.args = args
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.dy = dy if dy is not None else np.ones_like(y_vals)
        # self.fit_result = None
        self.__fit()
        if self.args is not None:
            self.fitted_fun = lambda x: self.func(self.fit_result.x, x, self.args)
        else:
            self.fitted_fun = lambda x: self.func(self.fit_result.x, x)

    def __cost(self, param):
        if self.args is not None:
            func_y = self.func(param, self.x_vals, self.args)
        else:
            func_y = self.func(param, self.x_vals)
        return ((self.y_vals - func_y)**2 / self.dy**2).sum()
    
    def __fit(self):
        self.fit_result : opt.OptimizeResult = opt.minimize(self.__cost, self.p0)
        print(self.fit_result)

    def eval(self, x):
        return self.fitted_fun(x)
    

class SOPlkTF:
    def __init__(
        self,
        pxp_container : PSContainer,
        sxp_container : PSContainer,
        bins : nmt.NmtBin,
        lmin = 30,
        lmax = 300,
        plk_frq = np.array([100, 143, 217, 353]),
        plk_dust_eff_frq = np.array([105.25, 148.235, 229.097, 372.193]),
        so_freq = np.array([90, 150])
    ):
        self.pxp_container : PSContainer = pxp_container
        self.sxp_container : PSContainer = sxp_container
        self.bins = bins
        self.e_l = bins.get_effective_ells()
        self.e_dl2cl = 2 * np.pi / self.e_l / (self.e_l + 1)
        self.msk = (self.e_l > lmin) * (self.e_l < lmax)
        self.msk_bin_ind = np.arange(self.msk.shape[0])[self.msk]

        self.planck_freqs = plk_frq
        self.planck_dust_eff_freqs = plk_dust_eff_frq
        self.so_freq = so_freq

        camb_dl = np.loadtxt("/scratch/gpfs/yl9946/basic_science/camb_lens_nobb.dat")
        camb_dl = np.concatenate([np.zeros(5, dtype=np.float64)[None, :], camb_dl], axis=1)
        self.cmb_ee = self.bins.bin_cell(camb_dl[:, 2]) * self.e_dl2cl
        self.cmb_bb = self.bins.bin_cell(camb_dl[:, 3]) * self.e_dl2cl