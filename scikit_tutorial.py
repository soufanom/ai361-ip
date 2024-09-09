import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform

# 1. Loading an Image
# First, load an image from a file or URL using skimage.io.
# Load the image from file or URL
image = io.imread('sample-images/palm-leaves-disease.png')

# Show the image
plt.imshow(image)
plt.axis('off')  # Hide axes
plt.show()

# 2. Manipulating Pixel Values Using NumPy
# You can modify the image by treating it as a NumPy array.
# Change the pixel values (e.g., set the top-left corner to white)
image[:100, :100] = [255, 255, 255, 255]  # Set top-left 100x100 pixels to white

# Display the modified image
plt.imshow(image)
plt.axis('off')
plt.show()

# 3. Changing Color Channels
# You can manipulate individual color channels (Red, Green, Blue) by accessing the respective channel using NumPy slicing.
# Set the red channel to zero (removes red from the image)
image[:, :, 0] = 0
image[:, :, 1] = 0

# Display the image with modified channels
plt.imshow(image)
plt.axis('off')
plt.show()

# 4. Convert Image to Grayscale
# scikit-image provides a function to convert RGB images to grayscale.
# Convert the image to grayscale
# Convert RGBA to RGB if the image has an alpha channel (4 channels)
if image.shape[2] == 4:
    image_rgb = color.rgba2rgb(image)
else:
    image_rgb = image  # Already in RGB

gray_image = color.rgb2gray(image_rgb)

# Display the grayscale image
plt.imshow(gray_image, cmap='gray')
plt.axis('off')
plt.show()

# 5. Resizing the Image
# You can resize the image using the transform.resize function.
# Resize the image to half of its original size
resized_image = transform.resize(image, (image.shape[0] // 2, image.shape[1] // 2))

# Display the resized image
plt.imshow(resized_image)
plt.axis('off')
plt.show()

# 6. Rotating the Image
# You can rotate the image using transform.rotate.
# Rotate the image by 45 degrees
rotated_image = transform.rotate(image, 45)

# Display the rotated image
plt.imshow(rotated_image)
plt.axis('off')
plt.show()

# 7. Flipping the Image
# To flip the image horizontally or vertically, you can use NumPy functions.
# Flip the image horizontally
flipped_image_h = np.fliplr(image)

# Flip the image vertically
flipped_image_v = np.flipud(image)

# Display the flipped images
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(flipped_image_h)
axs[0].axis('off')
axs[0].set_title('Horizontally Flipped')

axs[1].imshow(flipped_image_v)
axs[1].axis('off')
axs[1].set_title('Vertically Flipped')

plt.show()


