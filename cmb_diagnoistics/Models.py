from pygsm import trj2tcmb, planck_law


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
    param = (1/tf)
    """
    tf_fac = param[0]
    return 1/tf_fac * x