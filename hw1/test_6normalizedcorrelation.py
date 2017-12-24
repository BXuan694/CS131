import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from filters import zero_mean_cross_correlation
from filters import normalized_cross_correlation

temp = io.imread('template.jpg')
temp_grey = io.imread('template.jpg', as_grey=True)

# Load image
img = io.imread('shelf_dark.jpg')
img_grey = io.imread('shelf_dark.jpg', as_grey=True)

# Perform cross-correlation between the image and the template
out1 = zero_mean_cross_correlation(img_grey, temp_grey)
out2 = normalized_cross_correlation(img_grey, temp_grey)
# Find the location with maximum similarity
y1,x1 = (np.unravel_index(out1.argmax(), out1.shape))
y2,x2 = (np.unravel_index(out2.argmax(), out2.shape))

# Display image
plt.imshow(img)
plt.title('Result (red marker on the detected location)')
plt.axis('off')

# Draw marker at detcted location
plt.plot(x1, y1, 'rx', ms=25, mew=5)
plt.plot(x2, y2, 'bx', ms=25, mew=5)
plt.show()