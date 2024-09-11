import streamlit as st
import numpy as np
import cv2 # pip install opencv-python
from PIL import Image
import os


# Function to create noisy images
def add_noise(image, noise_level=0.2):
    noise = np.random.normal(loc=0, scale=noise_level, size=image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)  # Keep values between 0 and 1
    return noisy_image


# Function to average multiple noisy images
def average_images(images):
    return np.mean(images, axis=0)


# Main app function
def main(image_path):
    st.title("Image Noise Reduction using Averaging")

    # Check if file exists and read image
    if os.path.isfile(image_path):
        base_image = Image.open(image_path)
        base_image = np.array(base_image) / 255.0  # Normalize the image
        st.image(base_image, caption="Original Image", use_column_width=True)

        # Set number of noisy images to generate
        num_images = st.slider("Select number of noisy images to average", 1, 50, 10)

        # Generate noisy images
        noisy_images = [add_noise(base_image) for _ in range(num_images)]

        # Average the noisy images
        averaged_image = average_images(noisy_images)

        # Show noisy and averaged images
        st.subheader(f"Noisy Images (Number of images: {num_images})")
        st.image(noisy_images[-1], caption="One of the noisy images", use_column_width=True)

        st.subheader("Averaged Image")
        st.image(averaged_image, caption="Averaged Image", use_column_width=True)

        # Option to enhance the averaged image (e.g., contrast enhancement)
        enhance = st.checkbox("Enhance Averaged Image")
        if enhance:
            enhanced_image = cv2.convertScaleAbs(averaged_image * 255, alpha=1.5, beta=0)
            st.image(enhanced_image, caption="Enhanced Averaged Image", use_column_width=True)
    else:
        st.error("Image file not found. Please enter a valid path.")


if __name__ == "__main__":
    main(image_path="../sample-images/al-aqsa.jpeg")
