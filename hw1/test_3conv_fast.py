from filters import conv_fast
from filters import conv_nested
# Setup
import numpy as np
import matplotlib.pyplot as plt
from time import time
from skimage import io
from PIL import Image
# Open image as grayscale
img = io.imread('dog.jpg', as_grey=True)
plt.imshow(img)
plt.axis('off')
plt.title("Isn't he cute?")
plt.show()

kernel = np.array(
[
    [1,0,-1],
    [2,0,-2],
    [1,0,-1]
])

t0 = time()
out_fast = conv_fast(img, kernel)
t1 = time()
out_nested = conv_nested(img, kernel)
t2 = time()
#Image.fromarray(np.uint8(out_nested)).save("ne.jpg");
#Image.fromarray(np.uint8(out_fast)).save("fa.jpg");
# Compare the running time of the two implementations
print("conv_nested: took %f seconds." % (t2 - t1))
print("conv_fast: took %f seconds." % (t1 - t0))

# Plot conv_nested output
plt.subplot(1, 2, 1)
plt.imshow(out_nested)
plt.title('conv_nested')
plt.axis('off')

# Plot conv_fast output
plt.subplot(1, 2, 2)
plt.imshow(out_fast)
plt.title('conv_fast')
plt.axis('off')

# Make sure that the two outputs are the same
if not (np.max(out_fast - out_nested) < 1e-10):
    print("Different outputs! Check your implementation.")