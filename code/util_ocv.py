"""Useful functions with OpenCV.

Creation: 2019/08 rene.ebel@orange.com
"""

import cv2 as cv
import numpy as np

#------------------------------------------------------------------------------

def blend(img1, img2):
    """Blend img1 and img2 with img2 alpha layer and put the result in img1.

    img1.shape = h,w,3
    img2.shape = h,w,4
    """
    i1 = img1.astype(np.uint16)
    i2 = img2.astype(np.uint16)
    a2 = i2[:,:,3]
    a1 = 255 - a2
    for i in range(3):
       i1[:,:,i] = (a1*i1[:,:,i]+a2*i2[:,:,i])/255
    img1[:,:,:] = i1.astype(np.uint8)

#------------------------------------------------------------------------------

def replace(img1, img2, x, y, alpha=False):
    """Replace a subimage of img1 with img2.

    x, y (int or float): position of img2 in img1
    alpha: True for alpha-blending
        in this case: img1.shape = h1,w1,3 and img2.shape = h2,w2,4
    return: modified subimage coordinates (if needed)
    """
    p = (int(x), int(y))
    sa = img1.shape[1], img1.shape[0]
    sb = img2.shape[1], img2.shape[0]
    a0, b0, s = [0, 0], [0, 0], [0, 0]
    for k in range(2):
        if p[k] >= sa[k]:
            return
        a1 = p[k] + sb[k]
        if a1 <= 0:
            return
        if p[k] >= 0:
           a0[k] = p[k]
           b0[k] = 0
        else:
           a0[k] = 0
           b0[k] = - p[k]
        s[k] = (a1 if a1 <= sa[k] else sa[k]) - a0[k]
    if alpha:
        blend(img1[a0[1]:a0[1]+s[1], a0[0]:a0[0]+s[0]],
            img2[b0[1]:b0[1]+s[1], b0[0]:b0[0]+s[0]])
    else:
        img1[a0[1]:a0[1]+s[1], a0[0]:a0[0]+s[0]] = \
            img2[b0[1]:b0[1]+s[1], b0[0]:b0[0]+s[0]]
    return a0[0], a0[1], s[0], s[1]

#------------------------------------------------------------------------------
