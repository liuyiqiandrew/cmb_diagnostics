import cmb_diagnostics
import cmb_utils
import numpy as np
import os

def main():
    # create a box mask
    box1 = np.array(((-50,50),(-30,90))) * np.pi / 180 # clean CMB region south
    box2 = np.array(((-20,-165),(0,-130))) * np.pi / 180 # clean CMB region east
    mask1 = cmb_utils.maptools.carrbox2hpmask(512, box1)
    ap_m1 = cmb_utils.maptools.apodize_square_mask(mask1)
    mask2 = cmb_utils.maptools.carrbox2hpmask(512, box2)
    ap_m2 = cmb_utils.maptools.apodize_square_mask(mask2)
    mask = ap_m1 + ap_m2

    # transfer function from EE
    tf_ee_est = cmb_diagnostics.TransferFuncEE()
    tf_ee_est.init_mask(mask)
    tf_ee_est.calc_tf_ee()

    # transfer function from TE
    tf_te_est = cmb_diagnostics.TransferFuncTE()
    tf_te_est.init_mask(mask)
    tf_te_est.calc_tf_te()

    # polarization angle estimation from EB
    pa_est = cmb_diagnostics.PolAngEB()
    pa_est.init_mask(mask)
    pa_est.calc_pol_ang()

if __name__ == "__main__":
    main()