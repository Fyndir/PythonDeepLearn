"""Kernel convolutions."""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

# kernels
K1 = ((0, 0, 0), (0, 1, 0), (0, 0, 0))
K2 = ((1/9, 1/9, 1/9), (1/9, 1/9, 1/9), (1/9, 1/9, 1/9))
K3 = ((0, -1, 0), (-1, 5, -1), (0, -1, 0))
K4 = ((-1, -1, -1), (-1, 8, -1), (-1, -1, -1))

def imshow(img, pos, k, title, take_abs=False):
   k = np.array(k, np.float32)
   img_c = scipy.signal.convolve2d(img, k, mode='same')
   if take_abs:
       img_c = np.abs(img_c)
   plt.subplot(pos)
   plt.xticks([])
   plt.yticks([])
   plt.imshow(img_c, cmap='gray')
   plt.title(title)

convert_to_gray = lambda img: np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

img = convert_to_gray(plt.imread('../data/lena.png')).astype(np.float32)
imshow(img, 141, K1, 'identity')
imshow(img, 142, K2, 'box blur')
imshow(img, 143, K3, 'sharpen')
imshow(img, 144, K4, 'edge detection', True)
plt.subplots_adjust(left=0.01, right=0.99, wspace=0.05)
plt.show()
