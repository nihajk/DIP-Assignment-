import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to quantize the image to n levels using resizing
def quantize_with_resize(image, num_levels):
    # Determine the factor to quantize the image
    scale_factor = np.sqrt(256 / num_levels)

    # Resize the image down to a small size (based on the scale factor)
    small_image = cv2.resize(image, (0, 0), fx=1/scale_factor, fy=1/scale_factor, interpolation=cv2.INTER_LINEAR)

    # Resize the small image back to the original size
    quantized_image = cv2.resize(small_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    return quantized_image

# Load and convert the image to grayscale
image = cv2.imread('/Users/niharikajk/Desktop/Niha Assingment/Niha Assignment Q1/moon_image.jpeg', cv2.IMREAD_GRAYSCALE)

# Quantize the image to 32 grayscale levels
quantized_image = quantize_with_resize(image, num_levels=32)

# Display the original and quantized images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(quantized_image, cmap='gray')
plt.title('Quantized to 32 Levels')

plt.tight_layout()
plt.show()
