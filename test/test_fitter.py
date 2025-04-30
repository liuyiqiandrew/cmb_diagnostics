import numpy as np
from cmb_diagnostics.cmb_diagnoistics.Estimator import Fitter


def model(p, x):
    a, b = p
    return a * x + b


def main():
    x = np.arange(100)
    y = model((2, 0.5), x) + np.random.randn(100)
    lin_fitter = Fitter(model, (1, 1), x, y)
    print(lin_fitter.fit_result)
    print(lin_fitter.eval(10))


if __name__ == "__main__":
    main()