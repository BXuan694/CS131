from filters import conv_nested
from filters import conv_fast
import numpy as np
import matplotlib.pyplot as plt
from time import time
from skimage import io

img = io.imread('dog.jpg', as_grey=True)

kernel = np.array(
    [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ])

# Create a test image: a white square in the middle
test_img = np.zeros((9, 9))
test_img[3:6, 3:6] = 1

# Run your conv_nested function on the test image
#test_output = conv_nested(test_img, kernel)
test_output = conv_nested(test_img,kernel)
# Build the expected output
expected_output = np.zeros((9, 9))
expected_output[2:7, 2:7] = 1
expected_output[4, 2:7] = 2
expected_output[2:7, 4] = 2
expected_output[4, 4] = 4

# Plot the test image
plt.subplot(1, 3, 1)
plt.imshow(test_img)
plt.title('Test image')
plt.axis('off')

# Plot your convolved image
plt.subplot(1, 3, 2)
plt.imshow(test_output)
plt.title('Convolution')
plt.axis('off')

# Plot the exepected output
plt.subplot(1, 3, 3)
plt.imshow(expected_output)
plt.title('Exepected output')
plt.axis('off')
plt.show()

# Test if the output matches expected output
assert np.max(test_output -
              expected_output) < 1e-10, "Your solution is not correct."


# Simple convolution kernel.
# Feel free to change the kernel and to see different outputs.
kernel = np.array(
    [
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])

out = conv_nested(img, kernel)

# Plot original image
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

# Plot your convolved image
plt.subplot(2, 2, 3)
plt.imshow(out)
plt.title('Convolution')
plt.axis('off')

# Plot what you should get
solution_img = io.imread('convoluted_dog.jpg', as_grey=True)
plt.subplot(2, 2, 4)
plt.imshow(solution_img)
plt.title('What you should get')
plt.axis('off')


plt.show()
