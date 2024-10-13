import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('/Users/niharikajk/Desktop/Niha Assingment/Niha Assignment Q1/moon_image.jpeg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# The threshold value can be adjusted depending on the brightness of the moon.
_, binary_mask = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

# Erode to remove small noise
kernel = np.ones((5,5),np.uint8)
cleaned_mask = cv2.erode(binary_mask, kernel, iterations = 1)

# Display the original image and the mask
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cleaned_mask, cmap='gray')
plt.title('Binary Mask')

plt.show()
