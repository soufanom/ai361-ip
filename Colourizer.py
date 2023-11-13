import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from sklearn.cluster import KMeans


def colorize_image(image_path, colormap='hot'):
    # Load the grayscale image
    gray_image = io.imread(image_path, as_gray=True)

    # Normalize the image to range [0, 1]
    normalized_image = gray_image / gray_image.max()

    # Get the color map by name:
    cm = plt.get_cmap(colormap)

    # Apply the colormap like a function to any array:
    colored_image_input = cm(normalized_image)

    return gray_image, colored_image_input


def segment_and_colorize(image_path, n_segments=5, colormap='hot'):
    # Load image and flatten it
    image = io.imread(image_path, as_gray=True)
    image_flattened = image.reshape(-1, 1)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_segments, random_state=0).fit(image_flattened)
    labels = kmeans.labels_.reshape(image.shape)

    # Get the colormap
    cmap = plt.get_cmap(colormap, n_segments)

    # Create a colorized version of the image
    colorized_image = np.zeros((*image.shape, 3))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            colorized_image[i, j] = cmap(labels[i, j])[:3]  # Exclude alpha channel

    return colorized_image


def show_images_and_histograms(gray_image, colored_image):
    # Set up the matplotlib figure and axes
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Show the grayscale image
    axs[0, 0].imshow(gray_image, cmap='gray')
    axs[0, 0].set_title("Grayscale Image")
    axs[0, 0].axis('off')

    # Show the colored image
    axs[0, 1].imshow(colored_image)
    axs[0, 1].set_title("Colored Image")
    axs[0, 1].axis('off')

    # Display histogram for the grayscale image
    axs[1, 0].hist(gray_image.ravel(), bins=256, color='gray', alpha=0.7)
    axs[1, 0].set_title("Histogram of Grayscale Image")

    # Display histogram for the colored image
    # Extracting the RGB channels for histogram
    r = colored_image[:, :, 0]
    g = colored_image[:, :, 1]
    b = colored_image[:, :, 2]

    axs[1, 1].hist(r.ravel(), bins=256, color='red', alpha=0.7)
    axs[1, 1].hist(g.ravel(), bins=256, color='green', alpha=0.7)
    axs[1, 1].hist(b.ravel(), bins=256, color='blue', alpha=0.7)
    axs[1, 1].set_title("Histogram of Colored Image")

    plt.show()


def show_images(original_image_path, colorized_image):
    # Load the original grayscale image
    original_image = io.imread(original_image_path, as_gray=True)

    # Display the original and colorized images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original Grayscale Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(colorized_image)
    plt.title("Colorized Image")
    plt.axis('off')

    plt.show()


# Example usage
original_image_path = 'sample-images/xray-471.png'

colormap = 'CMRmap' # hot - plasma - Spectral -

gray_image, colored_image = colorize_image(original_image_path, colormap)
show_images_and_histograms(gray_image, colored_image)

colorized_image = segment_and_colorize(original_image_path, n_segments=4, colormap=colormap)
show_images(original_image_path, colorized_image)
