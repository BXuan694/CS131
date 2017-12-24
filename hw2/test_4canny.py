import matplotlib.pyplot as plt
from skimage import io
from edge import canny
import numpy as np

# Load image
img = io.imread('iguana.png', as_grey=True)

# Run Canny edge detector
edges = canny(img, kernel_size=5, sigma=1.4, high=0.03, low=0.02)

plt.imshow(edges,cmap='gray')
plt.axis('off')
plt.show()