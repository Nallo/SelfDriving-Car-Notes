import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

parser = argparse.ArgumentParser(description='Combining Sobel techniques.')
parser.add_argument('kernel_size',
                    type=int,
                    help='Sobel Kernel size - Must be an odd number')
args = parser.parse_args()

# Read in an image
image = mpimg.imread('signs_vehicles_xygrad.png')

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the derivative in x or y given orient = 'x' or 'y')
    sobel = np.zeros_like(gray)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    else:
        print("Cannot compute derivative along {} axis.".format(orient))
        return None

    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude
            # is > thresh_min and < thresh_max
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # 6) Return this mask as your binary_output image
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Calculate the magnitude as sqrt(dx^2 + dy^2)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*sobel_mag/np.max(sobel_mag))

    # 5) Create a binary mask where mag thresholds are met
    thresh_min = mag_thresh[0]
    thresh_max = mag_thresh[1]
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # 6) Return this mask as your binary_output image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the x and y gradients
    sobel_x_abs = np.absolute(sobel_x)
    sobel_y_abs = np.absolute(sobel_y)

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of
    # the gradient
    grad_direction = np.arctan2(sobel_y_abs, sobel_x_abs)

    # 5) Create a binary mask where direction thresholds are met
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    binary_output = np.zeros_like(grad_direction)
    binary_output[(grad_direction >= thresh_min) & (grad_direction <= thresh_max)] = 1

    # 6) Return this mask as your binary_output image
    return binary_output

# Choose a Sobel kernel size
ksize = args.kernel_size # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(20, 100))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.2))

# Finally combine all the results
combined = np.zeros_like(gradx)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

plt.figure()
plt.subplot(3, 2, 1)
plt.axis('off')
plt.title("gradx")
plt.imshow(gradx, cmap='gray')

plt.subplot(3, 2, 2)
plt.axis('off')
plt.title("grady")
plt.imshow(grady, cmap='gray')

plt.subplot(3, 2, 3)
plt.axis('off')
plt.title("mag_binary")
plt.imshow(mag_binary, cmap='gray')

plt.subplot(3, 2, 4)
plt.axis('off')
plt.title("dir_binary")
plt.imshow(dir_binary, cmap='gray')

plt.subplot(3, 2, 5)
plt.axis('off')
plt.title("combined")
plt.imshow(combined, cmap='gray')

plt.show()
