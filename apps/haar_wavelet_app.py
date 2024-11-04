import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pywt

def haar_wavelet_transform(image_array):
    # Apply Haar wavelet transform
    coeffs2 = pywt.dwt2(image_array, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return LL, LH, HL, HH

def normalize_component(component):
    # Normalize to the range 0-255
    component = cv2.normalize(component, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(component)

def main():
    st.title("Haar Wavelet Transform Viewer")
    st.write("Upload an image to see its Haar wavelet decomposition into LL, LH, HL, and HH parts.")

    # Upload Image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert uploaded file to a numpy array
        image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
        image_array = np.array(image)

        # Show original image
        st.image(image, caption="Original Image", use_column_width=True)

        # Perform Haar wavelet transform
        LL, LH, HL, HH = haar_wavelet_transform(image_array)

        # Normalize and display each component
        st.write("**LL (Approximation)**")
        st.image(normalize_component(LL), clamp=True, use_column_width=True)

        st.write("**LH (Vertical Details)**")
        st.image(normalize_component(LH), clamp=True, use_column_width=True)

        st.write("**HL (Horizontal Details)**")
        st.image(normalize_component(HL), clamp=True, use_column_width=True)

        st.write("**HH (Diagonal Details)**")
        st.image(normalize_component(HH), clamp=True, use_column_width=True)

if __name__ == "__main__":
    main()
