"""Adversarial perturbation of images (for GoogleNet classifier).

Reference:
    "Universal adversarial perturbations" (2017)
    https://arxiv.org/pdf/1610.08401.pdf
"""

import os

import cv2 as cv
import numpy as np

#------------------------------------------------------------------------------

# source and destination folder
IMG_FOLDER = '../data/photo_1'

# perturbation level (-LEVEL to +LEVEL)
LEVEL = 10

# GoogleNet classifier
IMG_PERTURB = '../data/perturb_googlenet.png'
SIZE = 224

#------------------------------------------------------------------------------

def resize_and_crop(img, size):

    h, w = img.shape[:2]
    if w == size and h == size:
        return img
    if w < h:
        w, h = size, int(size * h / w)
    else:
        w, h = int(size * w / h), size
    img = cv.resize(img, dsize=(w, h), interpolation=cv.INTER_AREA)

    x = round((w - size)/ 2)
    y = round((h - size)/ 2)
    return img[y:y+size, x:x+size].copy()

#------------------------------------------------------------------------------

def perturb_image(root, name, img_perturb):

    img = cv.imread(os.path.join(root, name))
    img = resize_and_crop(img, SIZE)  
    img = cv.add(img, img_perturb, dtype=cv.CV_8U)
    name, ext = os.path.splitext(name)
    name = os.path.join(root, 'perturb_' + name + '.png')
    cv.imwrite(name, img)

#------------------------------------------------------------------------------

def browse_and_perturb():

    img_perturb = np.asarray(cv.imread(IMG_PERTURB), np.float64)
    m = np.max(img_perturb) / 2
    img_perturb = np.around(LEVEL * (img_perturb / m - 1))
    img_perturb = np.asarray(img_perturb, np.int8)

    for root, dirs, files in os.walk(IMG_FOLDER):
        for name in files:
            perturb_image(root, name, img_perturb)

#------------------------------------------------------------------------------

browse_and_perturb()