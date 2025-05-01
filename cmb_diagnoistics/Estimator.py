import scipy.optimize as opt
import numpy as np
from .Container import PSContainer
from .Constants import PSType
from .Models import amp_dust_mbb
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
        beta_d = 1.6,
        T_d = 19.6,
        nu0 = 353.,
        so_freq = np.array([90, 150])
    ):
        self.pxp_container : PSContainer = pxp_container
        self.sxp_container : PSContainer = sxp_container
        assert(pxp_container.pstype == sxp_container.pstype)
        self.tftype : PSType = self.pxp_container.pstype
        self.bins = bins
        self.e_l = bins.get_effective_ells()
        self.e_dl2cl = 2 * np.pi / self.e_l / (self.e_l + 1)
        self.msk = (self.e_l > lmin) * (self.e_l < lmax)
        self.msk_bin_ind = np.arange(self.msk.shape[0])[self.msk]

        self.dust_eff_freq_lookup = {
            100: 105.25 ,
            143: 148.235,
            217: 229.097,
            353: 372.193,
        }
        self.so_freq = so_freq

        self.beta_d = beta_d
        self.T_d = T_d
        self.nu0 = nu0

        camb_dl = np.loadtxt("/scratch/gpfs/yl9946/basic_science/camb_lens_nobb.dat")
        camb_dl = np.concatenate([np.zeros(5, dtype=np.float64)[None, :], camb_dl], axis=1)
        match self.tftype:
            case PSType.PP:
                self.cmb_ee = self.bins.bin_cell(camb_dl[:, 2]) * self.e_dl2cl
                self.cmb_bb = self.bins.bin_cell(camb_dl[:, 3]) * self.e_dl2cl
            case PSType.TP:
                self.cmb_te = self.bins.bin_cell(camb_dl[:, 4]) * self.e_dl2cl
        
    def calc_tf(self):
        match self.tftype:
            case PSType.PP:
                self.__tf_ee()
            case PSType.TP:
                self.__tf_te()

    def __tf_ee(self):
        self.tf = np.zeros_like(self.msk_bin_ind, dtype=np.float64)
        # estimate dust first
        f1 = []
        f2 = []
        pxp_specs = []
        dpxp_specs = []
        for tcr in self.pxp_container.tracers['EE']:
            f1.append(self.dust_eff_freq_lookup[tcr[0]])
            f2.append(self.dust_eff_freq_lookup[tcr[1]])
            pxp, dpxp = self.pxp_container.get_spectrum('EE', tcr[0], tcr[1], return_ps_delta=True)
            pxp_specs.append(pxp[self.msk])
            dpxp_specs.append(dpxp[self.msk])
        f1, f2, pxp_specs, dpxp_specs = np.array(f1), np.array(f2), np.array(pxp_specs), np.array(dpxp_specs)

    def __est_dust_amp(self, f1, f2, pxp, dpxp):
        x_vals = (f1, f2)
        y_vals = pxp
        args = (self.beta_d, self.T_d, self.nu0)
        dust_fitter = Fitter(amp_dust_mbb, (1), x_vals, y_vals, args, dpxp)

        

    def __tf_te(self):
        pass