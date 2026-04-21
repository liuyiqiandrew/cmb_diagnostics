from pygsm import trj2tcmb, planck_law
import numpy as np

def amp_dust_mbb(params, x, args):
    """
    MBB with amplitude to be fitted

    params = (amp)
    x = (f1_arr, f2_arr)
    args = (beta_d, T_d, nu0)
    """
    amp = params[0]
    f1, f2 = x
    beta_d, T_d, nu0 = args

    r2c_f1 = trj2tcmb(f1)
    r2c_f2 = trj2tcmb(f2)
    mbb_pl = (f1 * f2 / nu0**2)**beta_d
    mbb_bb = planck_law(T_d, f1) * planck_law(T_d, f2) / planck_law(T_d, nu0)**2
    mbb_factor = mbb_pl * mbb_bb * r2c_f1 * r2c_f2
    return amp * mbb_factor


def tf_model(param, x):
    """
    param = tf
    """
    tf_fac = param[0]
    return tf_fac * x


def rttf_error(pxp_est, dpxs):
    """
    Calculate the error on the transfer function
    """
    info = rttf_fisher(pxp_est, dpxs)
    return 1 / np.sqrt(info)


def rttf_fisher(pxp_est, dpxs):
    """
    Calculate the Fisher matrix for the transfer function
    """
    return (pxp_est**2 / dpxs**2).sum()