"""Automatic image stitcher."""

import cv2 as cv

IMG_FOLDER = '../data/photo_pano/'
IMG_FILES = 'gfzhang_1.jpg', 'gfzhang_2.jpg', 'gfzhang_3.jpg'

imgs = [cv.imread(IMG_FOLDER + name) for name in IMG_FILES]
stitcher = cv.Stitcher_create()
retval, img_pano = stitcher.stitch(imgs)
cv.imshow('win', img_pano)
cv.waitKey(0)
cv.destroyAllWindows()
