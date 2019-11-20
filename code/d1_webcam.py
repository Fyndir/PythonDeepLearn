"""Show webcam feed, take snapshots."""

import cv2 as cv

# resolution
# 4/3:  (640, 480)  (800, 600)  (1280, 960)
# 16/9: (640, 360)  (800, 448)  (1280, 720)

RESOL = (1920, 1080)

def show_video():
    n = 1
    capture = cv.VideoCapture(0)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, RESOL[0])
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, RESOL[1])
    while True:
        ret, img = capture.read()
        cv.imshow('win', img)
        key = cv.waitKey(1)
        if key > 0:
            print(key)
        if key == 32:
            name = f'snap {n}.jpg'
            print(f'snapshot {name} {img.shape[1]} x {img.shape[0]}')
            params = [int(cv.IMWRITE_JPEG_QUALITY), 100]
            cv.imwrite(name, img, params)
            n += 1
        elif key == 27:
            break

    capture.release()
    cv.destroyAllWindows()

show_video()