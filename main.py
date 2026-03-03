import streamlit as st
import cv2
import numpy as np
from filters import *
from metrics import *
from visualization import *
from streamlit_image_comparison import image_comparison

st.set_page_config(layout="wide")
st.title("Interactive Image Noise Simulation & Restoration Tool")

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

    st.sidebar.header("Filter Settings")

    filter_type = st.sidebar.selectbox(
        "Select Filter",
        ["Mean", "Gaussian", "Median"]
    )

    ksize = st.sidebar.slider("Kernel Size (odd only)", 3, 15, 3, step=2)

    if filter_type == "Mean":
        restored = mean_filter(noisy, ksize)
    elif filter_type == "Gaussian":
        restored = gaussian_filter(noisy, ksize)
    else:
        restored = median_filter(noisy, ksize)

    st.subheader("Before & After Comparison")
    image_comparison(
        img1=cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB),
        img2=cv2.cvtColor(restored, cv2.COLOR_BGR2RGB),
        label1="Noisy",
        label2="Restored"
    )

    st.subheader("Quality Metrics")

    mse_value = calculate_mse(original, restored)
    psnr_value = calculate_psnr(original, restored)
    ssim_value = calculate_ssim(original, restored)

    col1, col2, col3 = st.columns(3)
    col1.metric("MSE", round(mse_value, 2))
    col2.metric("PSNR", round(psnr_value, 2))
    col3.metric("SSIM", round(ssim_value, 4))

    st.subheader("Histogram Analysis")

    col1, col2, col3 = st.columns(3)
    col1.pyplot(plot_histogram(original, "Original Histogram"))
    col2.pyplot(plot_histogram(noisy, "Noisy Histogram"))
    col3.pyplot(plot_histogram(restored, "Restored Histogram"))

    st.download_button(
        label="Download Restored Image",
        data=cv2.imencode(".png", restored)[1].tobytes(),
        file_name="restored_image.png",
        mime="image/png"
    )