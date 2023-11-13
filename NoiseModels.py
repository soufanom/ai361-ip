import numpy as np
import matplotlib.pyplot as plt
from skimage import io, util
from skimage.util import random_noise

def add_gaussian_noise(image, mean=1, var=0.01):
    return random_noise(image, mode='gaussian', mean=mean, var=var)

def add_salt_and_pepper_noise(image, salt_vs_pepper=1, amount=0.01):
    return random_noise(image, mode='s&p', salt_vs_pepper=salt_vs_pepper, amount=amount)

def add_rayleigh_noise(image, scale=0.25):
    # Rayleigh noise is not natively supported by skimage, so we implement it here
    rayleigh_noise = np.random.rayleigh(scale, image.shape)
    noisy_image = image + rayleigh_noise
    return np.clip(noisy_image, 0, 1)

def add_uniform_noise(image, low=0, high=0.25):
    uniform_noise = np.random.uniform(low, high, image.shape)
    noisy_image = image + uniform_noise
    return np.clip(noisy_image, 0, 1)

def add_exponential_noise(image, scale=0.35):
    # Exponential noise is not natively supported by skimage, so we implement it here
    exponential_noise = np.random.exponential(scale, image.shape)
    noisy_image = image + exponential_noise
    return np.clip(noisy_image, 0, 1)

def main(image_path):
    image = util.img_as_float(io.imread(image_path))

    # Apply different noises to the image
    gaussian_noisy = add_gaussian_noise(image)
    sp_noisy = add_salt_and_pepper_noise(image)
    rayleigh_noisy = add_rayleigh_noise(image)
    uniform_noisy = add_uniform_noise(image)
    exponential_noisy = add_exponential_noise(image)

    # Plot the noisy images
    fig, axes = plt.subplots(2, 3, figsize=(15, 10),
                             sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Original image')

    ax[1].imshow(gaussian_noisy, cmap=plt.cm.gray)
    ax[1].set_title('Gaussian noise')

    ax[2].imshow(sp_noisy, cmap=plt.cm.gray)
    ax[2].set_title('Salt and Pepper noise')

    ax[3].imshow(rayleigh_noisy, cmap=plt.cm.gray)
    ax[3].set_title('Rayleigh noise')

    ax[4].imshow(uniform_noisy, cmap=plt.cm.gray)
    ax[4].set_title('Uniform noise')

    ax[5].imshow(exponential_noisy, cmap=plt.cm.gray)
    ax[5].set_title('Exponential noise')

    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main('sample-images/al-aqsa.jpeg')
