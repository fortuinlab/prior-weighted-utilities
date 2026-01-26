import numpy as np


def gaussian_nll(y, mu, sd):
    var = sd**2
    return 0.5 * np.mean(((y - mu) ** 2) / var + np.log(2 * np.pi * var))
