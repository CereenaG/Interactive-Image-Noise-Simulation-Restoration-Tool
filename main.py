import streamlit as st
import cv2
import numpy as np
import pandas as pd
from filters import *
from metrics import *
from visualization import *
from streamlit_image_comparison import image_comparison

st.set_page_config(layout="wide")
st.title("Image Noise Simulation & Restoration Analysis Tool")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original = cv2.imdecode(file_bytes, 1)

    st.sidebar.header("Noise Settings")

    noise_type = st.sidebar.selectbox(
        "Select Noise Type",
        ["Salt & Pepper", "Gaussian"]
    )

    if noise_type == "Salt & Pepper":
        density = st.sidebar.slider("Noise Density", 0.0, 0.5, 0.1)
        noisy = add_salt_pepper_noise(original, density)
    else:
        std = st.sidebar.slider("Gaussian Std Dev", 1, 50, 10)
        noisy = add_gaussian_noise(original, 0, std)

    # ---- Apply All Filters ----
    mean_img = mean_filter(noisy, 3)
    gaussian_img = gaussian_filter(noisy, 5)
    median_img = median_filter(noisy, 3)

    # ---- Calculate Metrics ----
    data = []

    for name, img in [
        ("Mean Filter", mean_img),
        ("Gaussian Filter", gaussian_img),
        ("Median Filter", median_img)
    ]:
        mse_val = calculate_mse(original, img)
        psnr_val = calculate_psnr(original, img)
        ssim_val = calculate_ssim(original, img)

        data.append({
            "Filter": name,
            "MSE": round(mse_val, 2),
            "PSNR": round(psnr_val, 2),
            "SSIM": round(ssim_val, 4)
        })

    df = pd.DataFrame(data)

    # ---- Identify Best Filter ----
    best_filter = df.loc[df["SSIM"].idxmax()]["Filter"]

    if best_filter == "Mean Filter":
        best_image = mean_img
    elif best_filter == "Gaussian Filter":
        best_image = gaussian_img
    else:
        best_image = median_img

    # ======================================================
    # 🔥 SLIDER AT THE TOP (Noisy vs Best Restored)
    # ======================================================

    st.subheader("Before & After Comparison")

    image_comparison(
        img1=cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB),
        img2=cv2.cvtColor(best_image, cv2.COLOR_BGR2RGB),
        label1="Noisy",
        label2="Restored (Best Filter)"
    )

    # ---- Performance Table ----
    st.subheader("Performance Comparison")
    st.dataframe(df)

    st.success(f"Best Performing Filter (Based on SSIM): {best_filter}")

    # ---- Individual Filter Outputs ----
    st.subheader("Visual Comparison")

    col1, col2, col3 = st.columns(3)

    col1.image(cv2.cvtColor(mean_img, cv2.COLOR_BGR2RGB), caption="Mean Filter")
    col2.image(cv2.cvtColor(gaussian_img, cv2.COLOR_BGR2RGB), caption="Gaussian Filter")
    col3.image(cv2.cvtColor(median_img, cv2.COLOR_BGR2RGB), caption="Median Filter")

    # ---- Conclusion ----
    st.subheader("Analysis Conclusion")

    if noise_type == "Salt & Pepper":
        st.info(
            "For Salt and Pepper Noise, Median filtering provides superior restoration "
            "because it effectively removes impulse disturbances while preserving edges."
        )
    else:
        st.info(
            "For Gaussian Noise, linear filters such as Mean and Gaussian filtering "
            "perform effectively due to the statistical nature of Gaussian noise."
        )