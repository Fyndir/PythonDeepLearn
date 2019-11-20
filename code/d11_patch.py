"""Adversarial perturbation of images (for VGG16).

Reference:
    "Adversarial patch" (2018)
    https://arxiv.org/pdf/1712.09665.pdf
"""

import enum
import os

import cv2 as cv
import numpy as np

import util_ocv

#------------------------------------------------------------------------------

# source and destination folder
IMG_FOLDER = '../data/photo_1'

STICKER_SIZE = 70

# 'L': left, 'R': right, or integer
STICKER_POS_X = 'R'

# 'B': bottom, 'T': top, or integer
STICKER_POS_Y = 'T'

MARGIN = 5

# VGG16 classifier
IMG_STICKER = '../data/patch_vgg16.png'
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

def perturb_image(root, name, img_sticker, sticker_pos):

    img = cv.imread(os.path.join(root, name))
    img = resize_and_crop(img, SIZE)
    util_ocv.replace(img, img_sticker, x=sticker_pos[0], y=sticker_pos[1], alpha=True)
    name, ext = os.path.splitext(name)
    name = os.path.join(root, 'patch_' + name + '.png')
    cv.imwrite(name, img)

#------------------------------------------------------------------------------

def browse_and_perturb():

    img_sticker = cv.imread(IMG_STICKER, cv.IMREAD_UNCHANGED)
    img_sticker = cv.resize(img_sticker, dsize=(STICKER_SIZE, STICKER_SIZE),
        interpolation=cv.INTER_AREA)

    if STICKER_POS_X == 'L':
        x = MARGIN
    elif STICKER_POS_X == 'R':
        x = SIZE - STICKER_SIZE - MARGIN - 1
    else:
        x = STICKER_POS_X

    if STICKER_POS_Y == 'T':
        y = MARGIN
    elif STICKER_POS_Y == 'B':
        y = SIZE - STICKER_SIZE - MARGIN - 1
    else:
        y = STICKER_POS_Y

    for root, dirs, files in os.walk(IMG_FOLDER):
        for name in files:
            perturb_image(root, name, img_sticker, (x, y))

#------------------------------------------------------------------------------

browse_and_perturb()