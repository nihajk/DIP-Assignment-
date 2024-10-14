import cv2
import numpy as np
import matplotlib.pyplot as plt

# Kuwahara filter function
def kuwahara_filter(image, kernel_size=5):
    # Padding to avoid boundary issues
    padded_image = cv2.copyMakeBorder(image, kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2, cv2.BORDER_REFLECT)
    filtered_image = np.zeros_like(image)
    
    rows, cols = image.shape
    half_k = kernel_size // 2

    for i in range(rows):
        for j in range(cols):
            # Define the 4 sub-regions in the kernel window
            region1 = padded_image[i:i+half_k+1, j:j+half_k+1]  # Top-left
            region2 = padded_image[i:i+half_k+1, j+half_k:j+kernel_size]  # Top-right
            region3 = padded_image[i+half_k:i+kernel_size, j:j+half_k+1]  # Bottom-left
            region4 = padded_image[i+half_k:i+kernel_size, j+half_k:j+kernel_size]  # Bottom-right

            # Calculate the mean and variance for each region
            regions = [region1, region2, region3, region4]
            means = [np.mean(region) for region in regions]
            variances = [np.var(region) for region in regions]

            # Choose the region with the smallest variance and set the pixel to its mean
            min_variance_index = np.argmin(variances)
            filtered_image[i, j] = means[min_variance_index]

    return filtered_image

# Load and convert the image to grayscale
image = cv2.imread('moon_image.jpeg', cv2.IMREAD_GRAYSCALE)

# Apply the Kuwahara filter
kuwahara_filtered_image = kuwahara_filter(image, kernel_size=5)

# Display the original and Kuwahara filtered images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Grayscale Image')

plt.subplot(1, 2, 2)
plt.imshow(kuwahara_filtered_image, cmap='gray')
plt.title('Kuwahara Filtered Image')

plt.tight_layout()
plt.show()
