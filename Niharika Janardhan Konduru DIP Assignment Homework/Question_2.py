import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and convert the image to grayscale
image = cv2.imread('moon_image.jpeg', cv2.IMREAD_GRAYSCALE)

# Function for Floyd-Steinberg dithering
def floyd_steinberg_dithering(image):
    # Copy the image to avoid modifying the original
    dithered_image = np.copy(image).astype(np.float32)
    rows, cols = dithered_image.shape

    for y in range(rows):
        for x in range(cols):
            old_pixel = dithered_image[y, x]
            new_pixel = np.round(old_pixel / 255) * 255
            dithered_image[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            if x + 1 < cols:
                dithered_image[y, x + 1] += quant_error * 7 / 16
            if y + 1 < rows:
                if x - 1 >= 0:
                    dithered_image[y + 1, x - 1] += quant_error * 3 / 16
                dithered_image[y + 1, x] += quant_error * 5 / 16
                if x + 1 < cols:
                    dithered_image[y + 1, x + 1] += quant_error * 1 / 16

    return np.clip(dithered_image, 0, 255).astype(np.uint8)

# Function for Jarvis-Judice-Ninke dithering
def jarvis_judice_ninke_dithering(image):
    # Copy the image to avoid modifying the original
    dithered_image = np.copy(image).astype(np.float32)
    rows, cols = dithered_image.shape

    for y in range(rows):
        for x in range(cols):
            old_pixel = dithered_image[y, x]
            new_pixel = np.round(old_pixel / 255) * 255
            dithered_image[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            # Distribute the quantization error according to Jarvis-Judice-Ninke matrix
            if x + 1 < cols:
                dithered_image[y, x + 1] += quant_error * 7 / 48
            if x + 2 < cols:
                dithered_image[y, x + 2] += quant_error * 5 / 48

            if y + 1 < rows:
                if x - 2 >= 0:
                    dithered_image[y + 1, x - 2] += quant_error * 3 / 48
                if x - 1 >= 0:
                    dithered_image[y + 1, x - 1] += quant_error * 5 / 48
                dithered_image[y + 1, x] += quant_error * 7 / 48
                if x + 1 < cols:
                    dithered_image[y + 1, x + 1] += quant_error * 5 / 48
                if x + 2 < cols:
                    dithered_image[y + 1, x + 2] += quant_error * 3 / 48

            if y + 2 < rows:
                if x - 2 >= 0:
                    dithered_image[y + 2, x - 2] += quant_error * 1 / 48
                if x - 1 >= 0:
                    dithered_image[y + 2, x - 1] += quant_error * 3 / 48
                dithered_image[y + 2, x] += quant_error * 5 / 48
                if x + 1 < cols:
                    dithered_image[y + 2, x + 1] += quant_error * 3 / 48
                if x + 2 < cols:
                    dithered_image[y + 2, x + 2] += quant_error * 1 / 48

    return np.clip(dithered_image, 0, 255).astype(np.uint8)

# Apply Floyd-Steinberg Dithering
fs_dithered_image = floyd_steinberg_dithering(image)

# Apply Jarvis-Judice-Ninke Dithering
jjn_dithered_image = jarvis_judice_ninke_dithering(image)

# Display the results
plt.figure(figsize=(12, 8))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Grayscale')

plt.subplot(1, 3, 2)
plt.imshow(fs_dithered_image, cmap='gray')
plt.title('Floyd-Steinberg Dithering')

plt.subplot(1, 3, 3)
plt.imshow(jjn_dithered_image, cmap='gray')
plt.title('Jarvis-Judice-Ninke Dithering')

plt.tight_layout()
plt.show()
