import random
import numpy as np
from linalg import *
from imageManip import *
import matplotlib.pyplot as plt

image1_path = './image1.jpg'
image2_path = './image2.jpg'

def display(img):
    # Show image
    plt.imshow(img)
    plt.axis('off')
    plt.show()

image1 = load(image1_path)
image2 = load(image2_path)

display(image1)
display(image2)

new_image = change_value(image1)
display(new_image)
grey_image = convert_to_grey_scale(image1)
display(grey_image)

without_red = rgb_decomposition(image1, 'R')
without_blue = rgb_decomposition(image1, 'B')
without_green = rgb_decomposition(image1, 'G')

display(without_red)
display(without_blue)
display(without_green)

image_l = lab_decomposition(image1, 'L')
image_a = lab_decomposition(image1, 'A')
image_b = lab_decomposition(image1, 'B')

display(image_l)
display(image_a)
display(image_b)

image_h = hsv_decomposition(image1, 'H')
image_s = hsv_decomposition(image1, 'S')
image_v = hsv_decomposition(image1, 'V')

display(image_h)
display(image_s)
display(image_v)

image_mixed = mix_images(image1, image2, channel1='R', channel2='G')
display(image_mixed)

'''



x=im.load("./image1.jpg");
image1=im.load("./image1.jpg");
image2=im.load("./image2.jpg");
im.mix_images(image1, image2, "red", "green").save("mix.jpg");
print(np.shape(x));

y=im.change_value(x);
z=im.convert_to_grey_scale(x);
r=im.rgb_decomposition(x,"red");
s=im.rgb_decomposition(x,"green");
t=im.rgb_decomposition(x,"blue");
o=im.lab_decomposition(x,"L");
p=im.lab_decomposition(x,"A");
q=im.lab_decomposition(x,"B");

u=im.hsv_decomposition(x,"H");
v=im.hsv_decomposition(x,"S");
w=im.hsv_decomposition(x,"V");


y.save("y.jpg");
z.save("z.jpg");
r.save("r.jpg");
s.save("s.jpg");
t.save("t.jpg");

o.save("o.jpg");
p.save("p.jpg");
q.save("q.jpg");

u.save("u.jpg");
v.save("v.jpg");
w.save("w.jpg");'''