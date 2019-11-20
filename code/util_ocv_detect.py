"""Drawing of detected areas and labels.

Creation: 2019/08 rene.ebel@orange.com
"""

import cv2 as cv
from util_ocv_colors import COLOR

def show(img, rects, scale=False, xywh=False):
    """Show detected areas.

    img: image
    rects: list of rectangles (x0, y0, x1, y1) or
        rectangles with text (x0, y0, x1, y1, text)
    scale: True if coordinates lie in [0, 1] and must be scaled
    xywh: True if rectangles are described as (x, y, w, h)
    """
    RECT_THICKNESS = 2
    RECT_COLOR = COLOR.GREEN
    MARGIN = 10
    FONT = {
        'fontFace': cv.FONT_HERSHEY_SIMPLEX,
        'fontScale': 0.7,
        'thickness': 2}

    h, w = img.shape[:2]
    for rect in rects:
        x0, y0, x1, y1 = rect[:4]
        if xywh:
            x1, y1 = x0+x1, y0+y1
        if scale:
            x0, y0, x1, y1 = w*x0, h*y0, w*x1, h*y1
        to_int = lambda x: round(float(x))
        x0, y0, x1, y1 = to_int(x0), to_int(y0), to_int(x1), to_int(y1)
        cv.rectangle(img, (x0, y0), (x1, y1), RECT_COLOR, RECT_THICKNESS)
        if len(rect) < 5:
            continue
        text = rect[4]
        text_size = cv.getTextSize(text, **FONT)
        rect_width = text_size[0][0] + 2*MARGIN
        x2 = round((x0 + x1 - rect_width)/2)
        x3 = x2 + rect_width
        y2 = y1 + 4
        y3 = y2 + 29
        if y3 >= h:
            y3 = y0 - 4
            y2 = y3 - 29
        cv.rectangle(img, (x2, y2), (x3, y3), COLOR.WHITE, cv.FILLED)
        cv.putText(img, text, (x2+MARGIN, y2+21), color=COLOR.BLACK, **FONT)
