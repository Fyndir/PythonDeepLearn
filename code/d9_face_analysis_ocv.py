"""Age and gender classification using CNNs (Gil Levi, Tal Hassner, CVPR 2015).

References:
    https://talhassner.github.io/home/publication/2015_CVPR
    https://www.learnopencv.com/age-gender-classification-using-opencv-deep-learning-c-python
    https://github.com/spmallick/learnopencv/blob/master/AgeGender/AgeGender.py

"""

import os

import numpy as np
import cv2 as cv

import util_ocv_dnn
import util_ocv_detect

#------------------------------------------------------------------------------

IMG_FILE = '../data/photo_faces/1.jpg'

#------------------------------------------------------------------------------

MODEL_FACE = {
    'folder': '../data/models_caffe/face_detection',
    'model_file': 'res10_300x300_ssd_iter_140000.caffemodel',
    'config_file': 'res10.deploy.prototxt',
    'size': (300, 300),
    'mean': (104, 117, 123),
    'crop': False,
    'swap_rb': False
}

MODEL_GENDER = {
    'folder': '../data/models_caffe/face_analysis',
    'model_file': 'gender_net.caffemodel',
    'config_file': 'gender.deploy.prototxt',
    'size': (227, 227),
    'mean': (78.426338, 87.768914, 114.895848),
    'crop': False,
    'swap_rb': False,
    'labels': ['M', 'F'],
    'n_top': 2
}

MODEL_AGE = {
    'folder': '../data/models_caffe/face_analysis',
    'model_file': 'age_net.caffemodel',
    'config_file': 'age.deploy.prototxt',
    'size': (227, 227),
    'mean': (78.426338, 87.768914, 114.895848),
    'crop': False,
    'swap_rb': False,
    'labels': ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60+'],
    'n_top': 8
}

#------------------------------------------------------------------------------

def detect_and_analyse_faces(img, model_face, model_gender, model_age):

    CONFIDENCE_THRESHOLD = 0.4
    pred = model_face.predict(img)
    h, w = img.shape[:2]
    rects = []
    PADDING = 20
    for i in range(pred.shape[2]):
        confidence, x0, y0, x1, y1 = pred[0, 0, i, 2:7]
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        to_int = lambda x: round(float(x))
        x0, y0, x1, y1 = to_int(w*x0), to_int(h*y0), to_int(w*x1), to_int(h*y1)
        x0 = max(0, x0-PADDING)
        y0 = max(0, y0-PADDING)
        x1 = min(w-1, x1+PADDING)
        y1 = min(h-1, y1+PADDING)
        roi = img[y0:y1, x0:x1, :]       
        gender = model_gender.predict(roi)[0][1]
        age = model_age.predict(roi)[0][1]
        text = f'{gender} {age}'
        rects.append((x0, y0, x1, y1, text))
    util_ocv_detect.show(img, rects)

#------------------------------------------------------------------------------

def main():

    img = cv.imread(IMG_FILE)
    try:
        model_face = util_ocv_dnn.Model(MODEL_FACE)
        model_gender = util_ocv_dnn.Model(MODEL_GENDER)
        model_age = util_ocv_dnn.Model(MODEL_AGE)
    except FileNotFoundError:
        return
    detect_and_analyse_faces(img, model_face, model_gender, model_age)
    cv.imshow('win', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

#------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
