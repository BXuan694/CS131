import matplotlib.pyplot as plt
from skimage import io
from filters import zero_pad

img = io.imread('dog.jpg', as_grey=True)

# Show image
plt.imshow(img)
plt.axis('off')
plt.title("Isn't he cute?")
plt.show()

pad_width = 20  # width of the padding on the left and right
pad_height = 40  # height of the padding on the top and bottom

padded_img = zero_pad(img, pad_height, pad_width)

# Plot your padded dog
plt.subplot(1, 2, 1)
plt.imshow(padded_img)
plt.title('Padded dog')
plt.axis('off')

# Plot what you should get
solution_img = io.imread('padded_dog.jpg', as_grey=True)
plt.subplot(1, 2, 2)
plt.imshow(solution_img)
plt.title('What you should get')
plt.axis('off')

plt.show()
