import numpy as np
import healpy as hp
from pixell import enmap, reproject
import pymaster as nmt
import pygsm

def box2hpmask(nside, box):
    """
    Convert carr box to square mask
    """
    pix_ind = np.arange(hp.nside2npix(nside))
    pix_ang = np.array(hp.pix2ang(nside, pix_ind, lonlat=True)) / 180 * np.pi
    if box[1,1] < 0:
        ra_cut = (pix_ang[0] > (2 * np.pi + box[0, 1])) * (pix_ang[0] < (2 * np.pi + box[1, 1]))
    else:
        ra_cut = (pix_ang[0] > box[0, 1]) * (pix_ang[0] < box[1, 1])
    dec_cut = (pix_ang[1] > box[0, 0]) * (pix_ang[1] < box[1, 0])
    return ra_cut * dec_cut


def apodize_square_mask(mask):
    ZERO = 1e-4
    nhg = hp.smoothing(mask, 4 / 180 * np.pi)
    nhg[nhg < 0] = 0
    nhg /= nhg.max()
    tmp_mask = nhg > ZERO
    ap_mask = nmt.mask_apodization(tmp_mask, 10, "C2")
    return ap_mask


def read_carr2healpix(carr_path):
    carr_map = enmap.read_fits(carr_path)
    hpx_map = reproject.map2healpix(carr_map, method='spline', order=1)
    return hpx_map


def knox_covar(cl13, cl24, cl14, cl23, fsky, e_l, bin_wth):
    """
    Calculate covariance according to Knox Formula
    """
    return (cl13 * cl24 + cl14 * cl23) / (2 * e_l + 1) / fsky / bin_wth


def log_gauss_like(data, model, var):
    return (- (data - model)**2 / var).sum()


def dust_dl(a, f1, f2, beta):
    mbb1 = f1**beta * pygsm.planck_law(19.6, f1)
    mbb2 = f2**beta * pygsm.planck_law(19.6, f2)
    mbb_bench = 353.**beta * pygsm.planck_law(19.6, 353.)
    return a * mbb1 * mbb2 / (mbb_bench)**2


def dust_neg_lnlike(a, dust_unity, ee, data, var):
    model = dust_unity * a + ee
    return -log_gauss_like(data, model, var)


def tf_neg_lnlike(f, a, dust_unity, ee, data, var):
    model = f * (a * dust_unity + ee)
    return -log_gauss_like(data, model, var)


def hess_inv(dmdq, var):
    return 1 / (dmdq**2 / var).sum()


def res2ang(res):
    a = res.x[0]
    return np.arctan(a * 2) / 4


def pol_ang_cost(a, data, model_unity):
    return ((data - a * model_unity)**2).sum()


def neg_amp_lnlike(a, template, data, var):
    model = a * template
    return -log_gauss_like(data, model, var)