import numpy as np
import cv2


def add_salt_pepper_noise(image, density):
    noisy = image.copy()
    num_pixels = int(density * image.shape[0] * image.shape[1])

    # Salt
    coords = [np.random.randint(0, i - 1, num_pixels) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255

    # Pepper
    coords = [np.random.randint(0, i - 1, num_pixels) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0

    return noisy


def add_gaussian_noise(image, mean, std):
    image_float = image.astype(np.float32)
    gaussian = np.random.normal(mean, std, image.shape).astype(np.float32)

    noisy = image_float + gaussian
    noisy = np.clip(noisy, 0, 255)

    return noisy.astype(np.uint8)


def mean_filter(image, ksize):
    return cv2.blur(image, (ksize, ksize))


def gaussian_filter(image, ksize):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def median_filter(image, ksize):
    return cv2.medianBlur(image, ksize)