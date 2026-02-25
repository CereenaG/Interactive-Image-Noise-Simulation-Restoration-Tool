import numpy as np


def mse(original, restored):
    return np.mean((original - restored) ** 2)


def psnr(original, restored):
    m = mse(original, restored)

    if m == 0:
        return 100

    return 20 * np.log10(255.0 / np.sqrt(m))