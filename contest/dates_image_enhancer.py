import os
import cv2
import numpy as np


# Function to apply basic image enhancements
def enhance_image(image):
    # 1. Histogram Equalization to improve contrast
    equalized_image = cv2.equalizeHist(image)

    # 2. Gaussian Blur to smooth the image and reduce noise
    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)

    # 3. Sharpen the image using a kernel
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
    sharpened_image = cv2.filter2D(blurred_image, -1, sharpen_kernel)

    return sharpened_image


# Function to detect object size using contours
def detect_object_size(image):
    # Apply binary threshold to separate object from background
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours to detect the object
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If a contour is found, calculate the area of the largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        object_size = cv2.contourArea(largest_contour)
    else:
        object_size = 0  # No object detected

    return object_size


# Function to calculate the average size and standard deviation of object sizes
def calculate_size_statistics(sizes):
    avg_size = np.mean(sizes)
    std_size = np.std(sizes)
    return avg_size, std_size


# Function to group images based on object size
def group_images_by_size(sizes, avg_size, std_size):
    labels = []
    for size in sizes:
        if size < avg_size - std_size:  # Smaller than average (small)
            labels.append(0)
        elif avg_size - std_size <= size < avg_size:  # Around average (medium)
            labels.append(1)
        elif avg_size <= size < avg_size + std_size:  # Slightly larger than average (large)
            labels.append(2)
        else:  # Much larger than average (extra large)
            labels.append(3)
    return labels


# Function to create a color overlay for each group
def create_color_overlay(image, group):
    # Create an empty image with the same size but in color (BGR)
    color_overlay = np.zeros_like(image)
    color_overlay = cv2.cvtColor(color_overlay, cv2.COLOR_GRAY2BGR)

    # Apply different colors based on the group (total 4 groups)
    if group == 0:
        color_overlay[:] = (255, 0, 0)  # Blue for small
    elif group == 1:
        color_overlay[:] = (0, 255, 0)  # Green for medium
    elif group == 2:
        color_overlay[:] = (0, 0, 255)  # Red for large
    elif group == 3:
        color_overlay[:] = (255, 255, 0)  # Cyan for extra large

    return color_overlay


# Function to blend the color overlay with the original grayscale image
def blend_with_original(image, color_overlay):
    # Convert grayscale image to BGR format
    original_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Blend the original image with the color overlay (keeping the original intensity)
    blended_image = cv2.addWeighted(original_color, 0.7, color_overlay, 0.3, 0)

    return blended_image


# Function to resize image to reduce storage space
def resize_image(image, width):
    aspect_ratio = image.shape[1] / image.shape[0]  # Calculate aspect ratio
    new_height = int(width / aspect_ratio)  # Calculate new height to maintain aspect ratio
    resized_image = cv2.resize(image, (width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image


# Function to process and enhance images in a folder with resizing and compression
def process_images(input_folder, output_folder, resize_width=500, jpeg_quality=90):
    # Read all images
    images = []
    sizes = []
    filenames = []
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            img_path = os.path.join(input_folder, filename)
            gray_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray_image is not None:
                # Apply basic enhancements before size detection
                enhanced_image = enhance_image(gray_image)

                # Resize image to reduce storage space
                resized_image = resize_image(enhanced_image, resize_width)
                images.append(resized_image)
                filenames.append(filename)

                # Detect object size and store the size
                size = detect_object_size(resized_image)
                sizes.append(size)

    # Calculate average size and standard deviation
    avg_size, std_size = calculate_size_statistics(sizes)

    # Group images based on object size
    labels = group_images_by_size(sizes, avg_size, std_size)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Enhance and save images based on their group
    for image, label, filename in zip(images, labels, filenames):
        color_overlay = create_color_overlay(image, label)  # Create the color overlay
        blended_image = blend_with_original(image, color_overlay)  # Blend with the original image

        # Create output path
        output_path = os.path.join(output_folder, f"enhanced_{filename}")

        # Save with compression for JPEG images
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            cv2.imwrite(output_path, blended_image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        else:
            cv2.imwrite(output_path, blended_image)

        print(f"Saved enhanced image {output_path}")


# Example usage
input_folder = '/Users/soufanom/PycharmProjects/contest-ip-prep/dates-contest-images/'
output_folder = '/Users/soufanom/PycharmProjects/contest-ip-prep/dc-sample-submission/'

process_images(input_folder, output_folder)