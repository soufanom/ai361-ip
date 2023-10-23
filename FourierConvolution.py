import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, util
from scipy.ndimage import convolve

''' 
    Approach:
    1. Compute the DFT of the image.
    2. Create a frequency response that represents the desired convolution operation.
    3. Multiply the DFT of the image with the frequency response.
    4. Compute the inverse DFT of the result.
'''


def compute_dft(image):
    """Compute the DFT and shift the zero-frequency component to the center."""
    return np.fft.fftshift(np.fft.fft2(image))


def compute_idft(f_transform):
    """Inverse FFT shift and compute the inverse FFT."""
    return np.abs(np.fft.ifft2(np.fft.ifftshift(f_transform)))


# Load the image
img = io.imread('sample-images/xray-471.png')

# Convert to grayscale if it's an RGB image
if len(img.shape) == 3 and img.shape[2] == 3:
    img = util.img_as_float(color.rgb2gray(img))

# Define a simple 3x3 averaging filter
filter_ = np.array([[1 / 9, 1 / 9, 1 / 9],
                    [1 / 9, 1 / 9, 1 / 9],
                    [1 / 9, 1 / 9, 1 / 9]])

# Convolve the image with the filter in the spatial domain using scipy.ndimage.convolve
convolved_img = convolve(img, filter_)

# Compute DFT of the image
img_freq = compute_dft(img)

# Create a frequency response for the filter
filter_freq = np.fft.fftshift(np.fft.fft2(filter_, s=img.shape))

# Multiply the DFTs
multiplied_freq = img_freq * filter_freq

# Compute the inverse DFT to get the result in the spatial domain
result_img = compute_idft(multiplied_freq)

# Display the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(convolved_img, cmap='gray')
plt.title('Convolved Image (Spatial Domain)')

plt.subplot(1, 3, 3)
plt.imshow(result_img, cmap='gray')
plt.title('Result Image (Frequency Domain)')

plt.tight_layout()
plt.show()
