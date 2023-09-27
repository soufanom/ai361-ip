import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters
from skimage import io, filters, util, exposure
from scipy.ndimage import convolve


def threshold_image(image_path, threshold_value=None):
    # Load the image
    image = io.imread(image_path, as_gray=True)

    # If no threshold_value is provided, use Otsu's method to determine it
    if threshold_value is None:
        threshold_value = filters.threshold_otsu(image)

    # Apply thresholding
    binary_image = image > threshold_value

    return image, binary_image


def generate_negative(image):
    # Convert the image to 8-bit unsigned byte format
    image_byte = util.img_as_ubyte(image)
    # Generate the negative
    negative_image = 255 - image_byte
    return negative_image


def contrast_stretching(image):
    # Stretch the contrast of the image
    p2, p98 = np.percentile(image, (2, 98))
    stretched_image = exposure.rescale_intensity(image, in_range=(p2, p98))
    return stretched_image


'''
To apply an averaging filter to an image, we can use convolution with a kernel (or mask) 
that has all values set to 1 and then divides by the number of elements in the kernel. 
For a 3x3 averaging filter, the kernel would look like this:
1/9  1/9  1/9
1/9  1/9  1/9
1/9  1/9  1/9
'''


def apply_averaging_filter(image, size=3):
    # Define the averaging kernel
    kernel = np.ones((size, size)) / (size * size)
    # Apply the convolution
    filtered_image = convolve(image, kernel)
    return filtered_image


def view_matrix(original, thresholded):
    # Display the original and thresholded images
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(original, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(thresholded, cmap='gray')
    ax[1].set_title('Spatial Transformed Image')
    ax[1].axis('off')

    plt.show()


# Example usage
image_path = 'sample-images/xray-11.png'
original_image, thresholded_image = threshold_image(image_path)
view_matrix(original_image, thresholded_image)

negative_image = generate_negative(original_image)
view_matrix(original_image, negative_image)

stretched_image = contrast_stretching(original_image)
view_matrix(original_image, stretched_image)

averaged_image = apply_averaging_filter(original_image)
view_matrix(original_image, averaged_image)
