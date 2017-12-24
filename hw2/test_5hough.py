import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from edge import canny
from edge import hough_transform

# Load image
img = io.imread('road.jpg', as_grey=True)

# Run Canny edge detector
edges = canny(img, kernel_size=5, sigma=1.4, high=0.03, low=0.02)
'''
plt.subplot(211)
plt.imshow(img,cmap='gray')
plt.axis('off')
plt.title('Input Image')

plt.subplot(212)
plt.imshow(edges,cmap='gray')
plt.axis('off')
plt.title('Edges')
plt.show()
'''
H, W = img.shape

# Generate mask for ROI (Region of Interest)
mask = np.zeros((H, W))
for i in range(H):
    for j in range(W):
        if i > (H / W) * j and i > -(H / W) * j + H:
            mask[i, j] = 1

# Extract edges in ROI
roi = edges * mask
'''
plt.subplot(1,2,1)
plt.imshow(mask,cmap='gray')
plt.title('Mask')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(roi,cmap='gray')
plt.title('Edges in ROI')
plt.axis('off')
plt.show()
'''
# Perform Hough transform on the ROI
acc, rhos, thetas = hough_transform(roi)

print(acc.shape);
print(acc);

# Coordinates for right lane
xs_right = []
ys_right = []

# Coordinates for left lane
xs_left = []
ys_left = []

for i in range(20):
    idx = np.argmax(acc)
    r_idx = idx // acc.shape[1]
    t_idx = idx % acc.shape[1]
    acc[r_idx, t_idx] = 0  # Zero out the max value in accumulator

    rho = rhos[r_idx]
    theta = thetas[t_idx]

    # Transform a point in Hough space to a line in xy-space.
    a = - (np.cos(theta) / np.sin(theta))  # slope of the line
    b = (rho / np.sin(theta))  # y-intersect of the line

    # Break if both right and left lanes are detected
    if xs_right and xs_left:
        break

    if a < 0:  # Left lane
        if xs_left:
            continue
        xs = xs_left
        ys = ys_left
    else:  # Right Lane
        if xs_right:
            continue
        xs = xs_right
        ys = ys_right

    for x in range(img.shape[1]):
        y = a * x + b
        if y > img.shape[0] * 0.6 and y < img.shape[0]:
            xs.append(x)
            ys.append(int(round(y)))

plt.imshow(img,cmap='gray')
plt.plot(xs_left, ys_left, linewidth=5.0)
plt.plot(xs_right, ys_right, linewidth=5.0)
plt.axis('off')
plt.show()
