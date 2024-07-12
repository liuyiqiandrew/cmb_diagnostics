import numpy as np
import healpy as hp
from pixell import enmap, enplot, reproject, utils, curvedsky
import pymaster as nmt


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
    carr_map = enmap.read_fits(path)
    hpx_map = reproject.map2healpix(carr_map, method='spline', order=1)
    return hpx_map