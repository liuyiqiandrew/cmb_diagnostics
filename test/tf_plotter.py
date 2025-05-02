import numpy as np
import matplotlib.pyplot as plt

def main():
    tf_bf = np.load("bf_tf.npy")
    tf_ml = np.load("ml_tf.npy")

    plt.figure(dpi=300)

    e_l, tf90, tf150 = tf_bf
    plt.scatter(e_l, tf90, label='BF 090')
    plt.scatter(e_l, tf150, label='BF 150')

    e_l, tf90, tf150 = tf_ml
    plt.scatter(e_l[1:], tf90[1:], label='ML 090')
    plt.scatter(e_l[1:], tf150[1:], label='ML 150')
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"TF")
    plt.axhline(1, c='k', ls='--')
    plt.legend()
    plt.loglog()
    plt.savefig("TF_bf+ml_log.png")

if __name__ == "__main__":
    main()