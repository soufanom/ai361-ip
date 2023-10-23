import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, util
from skimage.filters import gaussian
from skimage.transform import rescale
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from skimage import exposure


def enhance_image(image):
    """Enhance the image using histogram equalization and contrast stretching."""
    # Histogram Equalization
    image_eq = exposure.equalize_hist(image)

    # Contrast Stretching
    p2, p98 = np.percentile(image_eq, (2, 98))
    image_contrast_stretched = exposure.rescale_intensity(image_eq, in_range=(p2, p98))

    return image_contrast_stretched


def fourier_transform(image):
    """Compute the 2D Fourier Transform of the given image."""
    f_transform = fft2(image)
    f_transform_centered = fftshift(f_transform)
    return f_transform_centered


def inverse_fourier_transform(f_transform_centered):
    """Compute the Inverse 2D Fourier Transform."""
    f_transform = ifftshift(f_transform_centered)
    image_reconstructed = ifft2(f_transform).real
    return image_reconstructed


def low_pass_filter(f_transform_centered, cutoff_frequency):
    """Apply a low-pass filter."""
    rows, cols = f_transform_centered.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 1
    return f_transform_centered * mask


def high_pass_filter(f_transform_centered, cutoff_frequency):
    """Apply a high-pass filter."""
    rows, cols = f_transform_centered.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 0
    return f_transform_centered * mask


def gaussian_low_pass_filter(f_transform_centered, sigma):
    """Apply a Gaussian low-pass filter."""
    rows, cols = f_transform_centered.shape
    crow, ccol = rows // 2, cols // 2
    x = np.linspace(-ccol, ccol, cols)
    y = np.linspace(-crow, crow, rows)
    x, y = np.meshgrid(x, y)
    mask = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return f_transform_centered * mask


def gaussian_high_pass_filter(f_transform_centered, sigma):
    """Apply a Gaussian high-pass filter."""
    return f_transform_centered * (1 - gaussian_low_pass_filter(f_transform_centered, sigma))


def main():
    # Load and preprocess the image
    img = io.imread('sample-images/Madinah-image-01.jpeg')
    # Convert to grayscale if it's an RGB image
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = util.img_as_float(color.rgb2gray(img))

    image_rescaled = rescale(img, 0.5, mode='reflect')
    image_enhanced = enhance_image(image_rescaled)
    img_enhanced = util.img_as_float(image_enhanced)

    # Fourier transform
    f_transform_centered = fourier_transform(img_enhanced)

    # Apply filters
    cutoff_frequency = 30
    lp_filtered = inverse_fourier_transform(low_pass_filter(f_transform_centered, cutoff_frequency))
    hp_filtered = inverse_fourier_transform(high_pass_filter(f_transform_centered, cutoff_frequency))
    glp_filtered = inverse_fourier_transform(gaussian_low_pass_filter(f_transform_centered, cutoff_frequency))
    ghp_filtered = inverse_fourier_transform(gaussian_high_pass_filter(f_transform_centered, cutoff_frequency))

    # Display results
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    ax[0][0].imshow(img, cmap='gray'), ax[0][0].set_title('Original Image')
    ax[0][1].imshow(img_enhanced, cmap='gray'), ax[0][1].set_title('Enhanced Image')
    ax[0][2].imshow(lp_filtered, cmap='gray'), ax[0][2].set_title('Low Pass Filtered')
    ax[1][0].imshow(hp_filtered, cmap='gray'), ax[1][0].set_title('High Pass Filtered')
    ax[1][1].imshow(glp_filtered, cmap='gray'), ax[1][1].set_title('Gaussian Low Pass Filtered')
    ax[1][2].imshow(ghp_filtered, cmap='gray'), ax[1][2].set_title('Gaussian High Pass Filtered')
    # for a in ax:
    #     a.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
