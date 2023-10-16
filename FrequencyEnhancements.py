import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import rescale


def plot_three_images(original, frequency, transformed):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(original, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(frequency, cmap='gray')
    ax[1].set_title('Frequency Domain')
    ax[1].axis('off')

    ax[2].imshow(transformed, cmap='gray')
    ax[2].set_title('Transformed Image')
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()


def transform_frequency(image_path):
    # Load image
    image = io.imread(image_path)

    # Check if the image is multi-channel (like RGB or RGBA)
    if len(image.shape) > 2:
        # Check number of channels and convert RGBA to RGB if necessary
        if image.shape[2] == 4:
            image = image[:, :, :3]

        # Convert to grayscale
        image = color.rgb2gray(image)

    # Compute 2D Fourier Transform and shift zero frequency to the center
    f_transform = np.fft.fft2(image)
    f_transform_centered = np.fft.fftshift(f_transform)

    # Convert to magnitude spectrum and apply logarithm for visualization
    magnitude_spectrum = np.log(np.abs(f_transform_centered) + 1)

    # Here, we'll just inverse transform without any modification
    # This will give us the original image back
    # But in practice, you can apply any frequency-domain operation before this step
    inverse_transform = np.abs(np.fft.ifft2(np.fft.ifftshift(f_transform_centered)))

    plot_three_images(image, magnitude_spectrum, inverse_transform)


def high_pass_filter(f_transform_centered, cutoff_frequency):
    rows, cols = f_transform_centered.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    # Create a mask with high values in the center (low frequency) and zero values outside
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 0

    # Apply mask to the shifted Fourier transform
    f_transform_centered_hp = f_transform_centered * mask

    return f_transform_centered_hp


def transform_frequency_with_high_pass(image_path, cutoff_frequency=30):
    # Load image
    image = io.imread(image_path)

    # Check if the image is multi-channel (like RGB or RGBA)
    if len(image.shape) > 2:
        # Check number of channels and convert RGBA to RGB if necessary
        if image.shape[2] == 4:
            image = image[:, :, :3]

        # Convert to grayscale
        image = color.rgb2gray(image)

    # Compute 2D Fourier Transform and shift zero frequency to the center
    f_transform = np.fft.fft2(image)
    f_transform_centered = np.fft.fftshift(f_transform)

    # Apply high-pass filter
    f_transform_centered_hp = high_pass_filter(f_transform_centered, cutoff_frequency)

    # Convert to magnitude spectrum for visualization
    magnitude_spectrum = np.log(np.abs(f_transform_centered) + 1)
    magnitude_spectrum_hp = np.log(np.abs(f_transform_centered_hp) + 1)

    # Inverse Fourier Transform for the high-pass filtered image
    inverse_transform_hp = np.abs(np.fft.ifft2(np.fft.ifftshift(f_transform_centered_hp)))

    plot_three_images(image, magnitude_spectrum_hp, inverse_transform_hp)


image_path = 'sample-images/Madinah-image-01.jpeg'  # Modify this
transform_frequency(image_path)
transform_frequency_with_high_pass(image_path, 1)


