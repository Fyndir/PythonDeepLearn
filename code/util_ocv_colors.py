"""A few colors in OpenCV format (B-G-R order).

References:
  https://github.com/opencv/opencv_contrib/blob/master/modules/viz/include/opencv2/viz/types.hpp
  https://docs.opencv.org/3.4/d4/dba/classcv_1_1viz_1_1Color.html
    
Creation: 2019/08 rene.ebel@orange.com
"""

from types import SimpleNamespace as SN

COLOR = SN(
    BLACK = (0, 0, 0),
    GRAY = (128, 128, 128),
    SILVER = (192, 192, 192),
    WHITE = (255, 255, 255),
    YELLOW = (0, 255, 255),
    GOLD = (0, 215, 255),
    ORANGE = (0, 165, 255),
    ORANGE_RED = (0, 69, 255),
    RED = (0, 0, 255),
    RASPBERRY = (92, 11, 227),
    CHERRY = (99, 29, 222),
    ROSE = (128, 0, 255),
    PINK = (203, 192, 255),
    APRICOT = (177, 206, 251),
    BROWN = (42, 42, 165),
    MAROON = (0, 0, 128),
    OLIVE = (0, 128, 128),
    GREEN = (0, 255, 0),
    CHARTREUSE = (0, 255, 128),
    LIME = (0, 255, 191),
    TEAL = (128, 128, 0),
    TURQUOISE = (208, 224, 64),
    CYAN = (255, 255, 0),
    BLUE = (255, 0, 0),
    AZURE = (255, 128, 0),
    CELESTIAL_BLUE = (208, 151, 73),
    BLUBERRY = (247, 134, 79),
    NAVY = (128, 0, 0),
    MAGENTA = (255, 0, 255),
    PURPLE = (128, 0, 128),
    VIOLET = (226, 43, 138),
    AMETHYST = (204, 102, 153),
    INDIGO = (130, 0, 75)
)

if __name__ == '__main__':

    import math
    import cv2 as cv
    import numpy as np

    MARGIN = 10
    RECT_SIZE = 30
    COL_SIZE = 250
    N_COLS = 2
    
    colors = COLOR.__dict__
    n_rows = math.ceil(len(colors) / N_COLS)
    h = MARGIN + n_rows * (RECT_SIZE + MARGIN)
    img = np.full((h, MARGIN + N_COLS*COL_SIZE, 3), 240, dtype=np.uint8)

    x0 = MARGIN
    y0 = MARGIN
    x1 = x0 + RECT_SIZE
    x2 = x1 + MARGIN
    row = 0
    for name in colors:
        y1 = y0 + RECT_SIZE
        cv.rectangle(img, (x0, y0), (x1, y1), colors[name], thickness=cv.FILLED)
        cv.putText(img, name, org=(x2, y0+23), fontScale=0.7, 
            fontFace=cv.FONT_HERSHEY_SIMPLEX, color=COLOR.BLACK, thickness=2)
        y0 = y1 + MARGIN
        row += 1
        if row == n_rows:
            row = 0
            x0 += COL_SIZE
            x1 = x0 + RECT_SIZE
            x2 = x1 + MARGIN
            y0 = MARGIN
    cv.imshow('colors', img)
    cv.waitKey(0)


