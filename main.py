import cv2
import os
import time

from filters import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    mean_filter,
    gaussian_filter,
    median_filter
)

from metrics import mse, psnr
from visualization import show_results


# -------- LOAD IMAGE --------

path = os.path.join("sample_images", "input.jpg")
image = cv2.imread(path)

if image is None:
    print("Image not found")
    exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# =====================================================
# EXPERIMENT 1 : GAUSSIAN NOISE
# =====================================================

gaussian_noise = add_gaussian_noise(image)

start = time.time()
mean_g = mean_filter(gaussian_noise)
t_mean_g = time.time() - start

start = time.time()
gaussian_g = gaussian_filter(gaussian_noise)
t_gauss_g = time.time() - start

start = time.time()
median_g = median_filter(gaussian_noise)
t_median_g = time.time() - start


print("\n===== Gaussian Noise Results =====")
print("Mean Filter PSNR:", psnr(image, mean_g))
print("Gaussian Filter PSNR:", psnr(image, gaussian_g))
print("Median Filter PSNR:", psnr(image, median_g))


show_results(
    "Gaussian Noise Removal Comparison",
    image,
    gaussian_noise,
    mean_g,
    gaussian_g,
    median_g
)


# =====================================================
# EXPERIMENT 2 : SALT & PEPPER NOISE
# =====================================================

sp_noise = add_salt_pepper_noise(image)

start = time.time()
mean_sp = mean_filter(sp_noise)
t_mean_sp = time.time() - start

start = time.time()
gaussian_sp = gaussian_filter(sp_noise)
t_gauss_sp = time.time() - start

start = time.time()
median_sp = median_filter(sp_noise)
t_median_sp = time.time() - start


print("\n===== Salt & Pepper Noise Results =====")
print("Mean Filter PSNR:", psnr(image, mean_sp))
print("Gaussian Filter PSNR:", psnr(image, gaussian_sp))
print("Median Filter PSNR:", psnr(image, median_sp))


show_results(
    "Salt & Pepper Noise Removal Comparison",
    image,
    sp_noise,
    mean_sp,
    gaussian_sp,
    median_sp
)

print("\n========== FINAL OBSERVATION ==========")

print("For Gaussian Noise:")
print("Linear smoothing filters perform effectively.")
print("Gaussian filter provides balanced noise reduction and detail preservation.")

print("\nFor Salt & Pepper Noise:")
print("Median filter performs best as it removes impulse noise efficiently.")