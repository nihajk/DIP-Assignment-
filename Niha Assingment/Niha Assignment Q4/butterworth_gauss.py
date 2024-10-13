import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to apply Fourier Transform
def fourier_transform(image):
    # Perform 2D Fourier Transform
    f_transform = np.fft.fft2(image)
    f_shifted = np.fft.fftshift(f_transform)  # Shift the zero frequency component to the center
    magnitude_spectrum = 20 * np.log(np.abs(f_shifted) + 1)  # For visualization
    return f_shifted, magnitude_spectrum

# Function to apply Gaussian low-pass filter in frequency domain
def gaussian_filter(shape, cutoff):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    x, y = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((x - center_row)**2 + (y - center_col)**2)
    gaussian_kernel = np.exp(-(dist_from_center**2) / (2 * (cutoff**2)))
    return gaussian_kernel

# Function to apply Butterworth low-pass filter in frequency domain
def butterworth_filter(shape, cutoff, order):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    x, y = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((x - center_row)**2 + (y - center_col)**2)
    butterworth_kernel = 1 / (1 + (dist_from_center / cutoff)**(2 * order))
    return butterworth_kernel

# Function to apply filter in the frequency domain and perform inverse transform
def apply_filter(f_shifted, filter_kernel):
    # Apply the filter to the Fourier-transformed image
    filtered_shifted = f_shifted * filter_kernel
    # Inverse Fourier transform to return to the spatial domain
    f_ishifted = np.fft.ifftshift(filtered_shifted)
    img_back = np.fft.ifft2(f_ishifted)
    img_back = np.abs(img_back)  # Taking the absolute value
    return img_back

# Load and convert the image to grayscale
image = cv2.imread('/Users/niharikajk/Desktop/Niha Assingment/Niha Assignment Q1/moon_image.jpeg', cv2.IMREAD_GRAYSCALE)

# Fourier Transform of the image
f_shifted, magnitude_spectrum = fourier_transform(image)

# Define cutoff frequency for the filters
cutoff = 30

# Apply Gaussian filter
gaussian_kernel = gaussian_filter(image.shape, cutoff)
gaussian_filtered_image = apply_filter(f_shifted, gaussian_kernel)

# Apply Butterworth filter with order 2
butterworth_kernel = butterworth_filter(image.shape, cutoff, order=2)
butterworth_filtered_image = apply_filter(f_shifted, butterworth_kernel)

# Display the results
plt.figure(figsize=(12, 10))

# Original Image and its Fourier Spectrum
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 3, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')

# Gaussian Filter and Filtered Image
plt.subplot(2, 3, 3)
plt.imshow(gaussian_kernel, cmap='gray')
plt.title('Gaussian Filter')

plt.subplot(2, 3, 4)
plt.imshow(gaussian_filtered_image, cmap='gray')
plt.title('Gaussian Filtered Image')

# Butterworth Filter and Filtered Image
plt.subplot(2, 3, 5)
plt.imshow(butterworth_kernel, cmap='gray')
plt.title('Butterworth Filter')

plt.subplot(2, 3, 6)
plt.imshow(butterworth_filtered_image, cmap='gray')
plt.title('Butterworth Filtered Image')

plt.tight_layout()
plt.show()
