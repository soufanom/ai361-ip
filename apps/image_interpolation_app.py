import streamlit as st
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# Function to generate a simple image (e.g., a 2x2 grayscale image)
def generate_simple_image():
    return np.array([[50, 100],
                     [150, 200]])

# Function to resize image using different interpolation methods
def resize_image(image, output_size, method):
    scale_factor = (output_size[0] / image.shape[0], output_size[1] / image.shape[1])
    if method == 'Nearest Neighbor':
        return ndimage.zoom(image, scale_factor, order=0)  # Nearest neighbor interpolation
    elif method == 'Bilinear':
        return ndimage.zoom(image, scale_factor, order=1)  # Bilinear interpolation
    elif method == 'Quadratic':
        return ndimage.zoom(image, scale_factor, order=2)  # Quadratic interpolation
    elif method == 'Cubic':
        return ndimage.zoom(image, scale_factor, order=3)  # Cubic interpolation
    else:
        return image  # Return the original image if no method is selected

# Function to plot the matrix and the image
def plot_image_and_matrix(image, resized_image):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the original image and its matrix
    axs[0].imshow(image, cmap='gray', interpolation='none')
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(resized_image, cmap='gray', interpolation='none')
    axs[1].set_title("Resized Image")
    axs[1].axis('off')

    return fig

# Streamlit app layout
st.title("Image Interpolation Demo")

# Select interpolation method
interpolation_method = st.selectbox(
    "Select Interpolation Method",
    ["Nearest Neighbor", "Bilinear", "Quadratic", "Cubic"]
)

# Select output image size
output_size = st.slider("Select Output Image Size (NxN):", 4, 16, 8, step=2)

# Generate original image
original_image = generate_simple_image()

# Resize image using the selected interpolation method and output size
resized_image = resize_image(original_image, (output_size, output_size), interpolation_method)

# Display original and resized images with their corresponding matrices
st.write("Original Image (2x2):")
st.write(original_image)

st.write(f"Resized Image ({output_size}x{output_size}) using {interpolation_method} Interpolation:")
st.write(np.round(resized_image, 2))

# Plot the images
fig = plot_image_and_matrix(original_image, resized_image)
st.pyplot(fig)
