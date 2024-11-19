import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift
from skimage.util import random_noise
from PIL import Image

# Function to add noise to an image
def add_noise(image, noise_type):
    if noise_type == "Gaussian":
        noisy_image = random_noise(image, mode='gaussian', var=0.01)
    elif noise_type == "Salt & Pepper":
        noisy_image = random_noise(image, mode='s&p', amount=0.05)
    elif noise_type == "Rayleigh":
        rayleigh_noise = np.random.rayleigh(scale=0.2, size=image.shape)
        noisy_image = np.clip(image + rayleigh_noise, 0, 1)
    else:
        noisy_image = image
    return (noisy_image * 255).astype(np.uint8)

# Function to compute Fourier Transform and generate spectrum
def compute_fourier_spectrum(image):
    fft_result = fftshift(fft2(image))
    spectrum = np.log(1 + np.abs(fft_result))
    return spectrum

# Streamlit UI
st.title("Image Noise and Fourier Frequency Analysis")

# Image upload
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image:
    # Load and preprocess image
    image = Image.open(uploaded_image).convert("L")
    image = np.array(image) / 255.0  # Normalize to [0, 1]

    # Select noise type
    noise_type = st.selectbox("Select Noise Type", ["None", "Gaussian", "Salt & Pepper", "Rayleigh"])

    # Add noise to the image
    noisy_image = add_noise(image, noise_type)

    # Compute Fourier spectrum for the original and noisy images
    original_spectrum = compute_fourier_spectrum(image)
    noisy_spectrum = compute_fourier_spectrum(noisy_image)

    # Display original image and its Fourier spectrum
    st.subheader("Original Image and Fourier Spectrum")
    col1, col2 = st.columns(2)
    with col1:
        st.image((image * 255).astype(np.uint8), caption="Original Image", use_column_width=True)
    with col2:
        plt.figure(figsize=(5, 5))
        plt.imshow(original_spectrum, cmap="gray")
        plt.title("Fourier Spectrum (Original)")
        plt.axis("off")
        st.pyplot(plt)

    # Display noisy image and its Fourier spectrum
    st.subheader("Noisy Image and Fourier Spectrum")
    col3, col4 = st.columns(2)
    with col3:
        st.image(noisy_image, caption=f"Image with {noise_type} Noise", use_column_width=True)
    with col4:
        plt.figure(figsize=(5, 5))
        plt.imshow(noisy_spectrum, cmap="gray")
        plt.title("Fourier Spectrum (Noisy)")
        plt.axis("off")
        st.pyplot(plt)
else:
    st.info("Please upload an image to begin.")
