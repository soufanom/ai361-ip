import numpy as np
import matplotlib.pyplot as plt
from skimage import io, util, color, exposure
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

def homomorphic_filter(img, gamma_low=0.3, gamma_high=1.5, cutoff=30):
    # Take the logarithm of the image to separate illumination and reflectance
    img_log = np.log1p(util.img_as_float(img))

    # Apply Fourier transform
    img_fft = fft2(img_log)
    img_fft_shifted = fftshift(img_fft)

    rows, cols = img.shape
    center_x, center_y = rows // 2, cols // 2
    filter = np.ones((rows, cols))

    for i in range(rows):
        for j in range(cols):
            dist = ((i - center_x) ** 2 + (j - center_y) ** 2)**0.5
            filter[i, j] = (gamma_high - gamma_low) * (1 - np.exp(-dist / cutoff)) + gamma_low

    # Apply the filter in the frequency domain
    filtered_fft = img_fft_shifted * filter
    filtered_fft_shifted_back = ifftshift(filtered_fft)

    # Inverse Fourier transform
    filtered_img_log = np.real(ifft2(filtered_fft_shifted_back))

    # Exponentiate the result to get back to the spatial domain
    filtered_img = np.expm1(filtered_img_log)

    return np.clip(filtered_img, 0, 1)


img = io.imread('sample-images/xray-471.png')
if len(img.shape) == 3:
    img = color.rgb2gray(img)

filtered_img = homomorphic_filter(img)
# Apply histogram equalization
eq_filtered_img = exposure.equalize_hist(filtered_img)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(filtered_img, cmap='gray')
ax[1].set_title('After Homomorphic Filtering')
ax[1].axis('off')

ax[2].imshow(eq_filtered_img, cmap='gray')
ax[2].set_title('After Histogram Equalization')
ax[2].axis('off')

plt.tight_layout()
plt.show()
