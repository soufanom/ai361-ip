from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

def number_to_pixel_matrix(num, image_size=(100, 100), font_size=40):
    """
    Convert a number into a matrix of pixels.

    Parameters:
        - num (int): The number to be converted.
        - image_size (tuple): Dimensions of the image to be generated.
        - font_size (int): Size of the font for the number.

    Returns:
        - numpy.ndarray: 2D array representing the grayscale pixel values of the number.
    """

    # Create a blank white image
    image = Image.new("L", image_size, color=255)

    # Draw the number on the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("resources/arial.ttf", font_size)
    text_width, text_height = draw.textsize(str(num), font=font)
    position = ((image_size[0] - text_width) / 2, (image_size[1] - text_height) / 2)
    draw.text(position, str(num), font=font, fill=0)

    # Convert the image to an array of pixel values
    pixel_matrix = np.array(image)

    return pixel_matrix


def view_matrix(matrix):
    """
    Visualize the matrix using matplotlib.

    Parameters:
        - matrix (numpy.ndarray): 2D array representing the grayscale pixel values.

    Returns:
        None
    """
    plt.imshow(matrix, cmap='gray')
    plt.axis('off')  # Turn off the axis numbers and ticks
    plt.show()

def modify_matrix(matrix, operation='invert'):
    """
    Modify the matrix based on specified operation.

    Parameters:
        - matrix (numpy.ndarray): 2D array representing the grayscale pixel values.
        - operation (str): The operation to be applied. One of ['invert', 'increase_brightness',
                           'decrease_brightness', 'increase_contrast'].

    Returns:
        numpy.ndarray: The modified matrix.
    """
    if operation == 'invert':
        return 255 - matrix
    elif operation == 'increase_brightness':
        return np.clip(matrix + 50, 0, 255)  # Add 50 to each pixel value but ensure they remain in [0, 255]
    elif operation == 'decrease_brightness':
        return np.clip(matrix - 50, 0, 255)  # Subtract 50 from each pixel value but ensure they remain in [0, 255]
    elif operation == 'increase_contrast':
        # Increase the contrast by increasing the difference from the midpoint (127.5)
        return np.clip(127.5 + (matrix - 127.5) * 1.5, 0, 255)
    else:
        raise ValueError(f"Unsupported operation: {operation}")

# Test the function
num = 2
matrix = number_to_pixel_matrix(num)
print(matrix)
view_matrix(matrix)

# Apply modification and view again
modified_matrix = modify_matrix(matrix, operation='increase_brightness')
view_matrix(modified_matrix)