import streamlit as st
import numpy as np
import cv2
from PIL import Image


def holomorphic_filtering(image, gamma_low, gamma_high, distance_threshold):
    """Apply holomorphic filtering to the input image."""
    # Convert to grayscale if not already
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Fourier Transform of the image
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    # Generate filter mask
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if d < distance_threshold:
                mask[i, j] = gamma_low
            else:
                mask[i, j] = gamma_high

    # Apply the filter
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Normalize the result to the range [0, 255]
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return img_back


# Streamlit App UI
st.title("Holomorphic Filtering Visualizer")

st.sidebar.header("Adjust Filter Parameters")
gamma_low = st.sidebar.slider("Gamma Low", 0.0, 1.0, 0.5, 0.01)
gamma_high = st.sidebar.slider("Gamma High", 0.0, 1.0, 1.0, 0.01)
distance_threshold = st.sidebar.slider("Distance Threshold", 1, 300, 50, 1)

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="Original Image", use_column_width=True)

    filtered_image = holomorphic_filtering(image, gamma_low, gamma_high, distance_threshold)

    st.image(filtered_image, caption="Filtered Image", use_column_width=True)

    st.write(f"**Gamma Low:** {gamma_low}")
    st.write(f"**Gamma High:** {gamma_high}")
    st.write(f"**Distance Threshold:** {distance_threshold}")
