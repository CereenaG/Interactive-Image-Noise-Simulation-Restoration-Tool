import cv2
import numpy as np


# ---------------- NOISE MODELS ----------------

def add_gaussian_noise(image, sigma=50):
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image, prob=0.08):
    noisy = image.copy()
    rand = np.random.rand(*image.shape[:2])

    noisy[rand < prob] = 0
    noisy[rand > 1 - prob] = 255

    return noisy


# ---------------- FILTERS ----------------

# Linear Filter
def mean_filter(image):
    return cv2.blur(image, (5, 5))


# Linear Filter
def gaussian_filter(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


# Non-Linear Filter
def median_filter(image):
    return cv2.medianBlur(image, 5)