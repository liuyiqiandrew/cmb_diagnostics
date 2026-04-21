import numpy as np
import matplotlib.pyplot as plt

def main():
    tf_bf = np.load("bf_tf.npy")
    tf_ml = np.load("ml_tf.npy")

    plt.figure(dpi=300)

    e_l, tf90, dtf90, tf150, dtf150 = tf_bf
    plt.errorbar(e_l, tf90, dtf90, label='BF 090',\
        ls='', marker='.', alpha=.5, capsize=3)
    plt.errorbar(e_l, tf150, dtf150, label='BF 150',\
        ls='', marker='.', alpha=.5, capsize=3)

    e_l, tf90, dtf90, tf150, dtf150 = tf_ml
    plt.errorbar(e_l[1:], tf90[1:] / 0.49, dtf90[1:] / .49, label='ML 090',\
        ls='', marker='.', alpha=.5, capsize=3)
    plt.errorbar(e_l[1:], tf150[1:] / 0.49, dtf150[1:] / .49, label='ML 150',\
        ls='', marker='.', alpha=.5, capsize=3)
    tf90 = np.load("/scratch/gpfs/yl9946/iso_maps/TF/transfer_function_SATp3_f090_south_science_x_SATp3_f090_south_science.npz")
    tf150 = np.load("/scratch/gpfs/yl9946/iso_maps/TF/transfer_function_SATp3_f150_south_science_x_SATp3_f150_south_science.npz")
    plt_el = np.arange(60) * 10 + 5.5
    plt.plot(plt_el, tf90['EE_to_EE'][:60], c='k', ls='--')
    plt.plot(plt_el, tf150['EE_to_EE'][:60], c='k', ls='--')
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"TF")
    plt.ylim(-0.2, 1.2)
    plt.axhline(1, c='k', ls='--')
    plt.legend()
    # plt.loglog()
    plt.savefig("TF_bf+ml.png")

if __name__ == "__main__":
    main()