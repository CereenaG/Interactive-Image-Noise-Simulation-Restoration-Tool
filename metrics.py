import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


def calculate_mse(original, restored):
    return np.mean((original - restored) ** 2)


def calculate_psnr(original, restored):
    mse = calculate_mse(original, restored)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(original, restored):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    restored_gray = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(original_gray, restored_gray, full=True)
    return score