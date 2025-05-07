import numpy as np
import matplotlib.pyplot as plt

def main():
    tf_bf = np.load("bf_tf.npy")
    tf_ml = np.load("ml_tf.npy")

    plt.figure(dpi=300)

    e_l, tf90, dtf90, tf150, dtf150 = tf_bf
    plt.errorbar(e_l, tf90, dtf90, label='BF 090',\
        ls='', marker='.')
    plt.errorbar(e_l, tf150, dtf150, label='BF 150',\
        ls='', marker='.')

    e_l, tf90, dtf90, tf150, dtf150 = tf_ml
    plt.errorbar(e_l[1:], tf90[1:] / 0.49, dtf90[1:] / .49, label='ML 090',\
        ls='', marker='.')
    plt.errorbar(e_l[1:], tf150[1:] / 0.49, dtf150[1:] / .49, label='ML 150',\
        ls='', marker='.')
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"TF")
    plt.ylim(-0.2, 1.2)
    plt.axhline(1, c='k', ls='--')
    plt.legend()
    # plt.loglog()
    plt.savefig("TF_bf+ml.png")

if __name__ == "__main__":
    main()