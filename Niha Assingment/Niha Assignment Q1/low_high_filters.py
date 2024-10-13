import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Load the image
image = cv2.imread('/Users/niharikajk/Desktop/Niha Assingment/Niha Assignment Q1/moon_image.jpeg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a low-pass Gaussian filter
gaussian_blur = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Apply a low-pass Average filter
average_blur = cv2.blur(gray_image, (5, 5))

# Apply a threshold to create a binary mask (Threshold values: 50 to 255)
_, binary_mask = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

# Apply a high-pass Laplacian filter
laplacian_filter = cv2.Laplacian(gray_image, cv2.CV_64F)

# Apply a high-pass Prewitt filter (using ndimage to define Prewitt kernels)
prewitt_kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
prewitt_kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
prewitt_x = ndimage.convolve(gray_image, prewitt_kernel_x)
prewitt_y = ndimage.convolve(gray_image, prewitt_kernel_y)
prewitt_filter = np.hypot(prewitt_x, prewitt_y)  # Magnitude of gradient

# Display the images
plt.figure(figsize=(12, 8))

# Original Image
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Binary Mask
plt.subplot(2, 3, 2)
plt.imshow(binary_mask, cmap='gray')
plt.title('Binary Mask (Threshold 50-255)')

# Gaussian Blur
plt.subplot(2, 3, 3)
plt.imshow(gaussian_blur, cmap='gray')
plt.title('Gaussian Blur')

# Average Blur
plt.subplot(2, 3, 4)
plt.imshow(average_blur, cmap='gray')
plt.title('Average Blur')

# Laplacian Filter
plt.subplot(2, 3, 5)
plt.imshow(np.abs(laplacian_filter), cmap='gray')
plt.title('Laplacian Filter')

# Prewitt Filter
plt.subplot(2, 3, 6)
plt.imshow(prewitt_filter, cmap='gray')
plt.title('Prewitt Filter')

plt.tight_layout()
plt.show()
