import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, segmentation, color, measure
from skimage.morphology import disk
from skimage.filters import rank
from skimage.util import img_as_ubyte
from scipy import ndimage as ndi

# Load an image
image_path = 'sample-images/xray-471.png'  # Replace with your image path
image = io.imread(image_path)

grayflag = False
# Check if image is already in grayscale
if len(image.shape) == 2 or image.shape[2] == 1:
    grayflag = True
    gray_image = image  # Image is already grayscale
elif image.shape[2] == 3:
    gray_image = color.rgb2gray(image)  # Convert to grayscale
else:
    raise ValueError("Image format not recognized. Expected a grayscale or RGB image.")

# 1. Thresholding
thresh = filters.threshold_otsu(gray_image)
binary_image = gray_image > 145

# 2. Watershed
# Compute elevation map for watershed
# denoise image
image_ws = img_as_ubyte(gray_image)
denoised = rank.median(image_ws, disk(2))
# find continuous region (low gradient -
# where less than 10 for this image) --> markers
# disk(5) is used here to get a more smooth image
markers = rank.gradient(denoised, disk(2)) < 10
markers = ndi.label(markers)[0]

# local gradient (disk(2) is used to keep edges thin)
gradient = rank.gradient(denoised, disk(2))

# process the watershed
segmentation_ws = segmentation.watershed(gradient, markers)

# 3. SLIC Superpixels
if grayflag:
    # Assuming gray_image is your grayscale image obtained from previous steps
    segmentation_slic = segmentation.slic(gray_image, n_segments=50, compactness=10, start_label=1, channel_axis=None)
else:
    segmentation_slic = segmentation.slic(image, n_segments=50, compactness=10, start_label=1)

# Display original and segmented images
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
ax = axes.ravel()

ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[0].axis('off')

# Get the color map by name:
cm = plt.get_cmap('gray')
ax[1].imshow(binary_image, cmap=cm)
ax[1].set_title('Thresholding')
ax[1].axis('off')

ax[2].imshow(segmentation.mark_boundaries(image, segmentation_ws))
ax[2].set_title('Watershed')
ax[2].axis('off')

ax[3].imshow(segmentation.mark_boundaries(image, segmentation_slic))
ax[3].set_title('SLIC Superpixels')
ax[3].axis('off')

plt.tight_layout()
plt.show()
