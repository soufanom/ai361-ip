"""
    Code generated using GPT-4
    Prompt used: Write a python code using scikit-image library to load an image and apply some filters to the image. Each time the code should show the image.
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, color
from scipy.ndimage import convolve

# Load the image
image_path = 'sample-images/Madinah-image-01.jpeg'  # Replace this with the path to your image
image = io.imread(image_path)

# Show the original image
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

# Apply Gaussian filter
gaussian_filtered = filters.gaussian(image, sigma=5)
plt.subplot(2, 2, 2)
plt.imshow(gaussian_filtered)
plt.title('Gaussian Filtered')
plt.axis('off')

# Apply Sobel filter
sobel_filtered = filters.sobel(color.rgb2gray(image))
plt.subplot(2, 2, 3)
plt.imshow(sobel_filtered, cmap='gray')
plt.title('Sobel Filtered')
plt.axis('off')

# Apply horizontal edge filter
horizontal_edge_filter = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])
horizontal_edges = convolve(color.rgb2gray(image), horizontal_edge_filter)
plt.subplot(2, 2, 4)
plt.imshow(horizontal_edges, cmap='gray')
plt.title('Horizontal Edges')
plt.axis('off')

# Display all images
plt.tight_layout()
plt.show()
