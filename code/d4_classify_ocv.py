"""Image classification with pretrained CNNs (OpenCV, Caffe models).

Creation: 2019-08 rene.ebel@orange.com
"""

import os

import numpy as np
import cv2 as cv

import util_ilsvrc
import util_ocv_dnn

#------------------------------------------------------------------------------

IMG_FOLDER = '../data/photo_1'

# 1: AlexNet
# 2: GoogleNet
# 3: VGG16
OPTION_MODEL = 1

N_TOP = 5

#------------------------------------------------------------------------------

MODELS = {
    1: {
        'name': 'AlexNet',
        'folder': '../data/models_caffe/classification/',
        'model_file': 'bvlc_alexnet.caffemodel',
        'config_file': 'bvlc_alexnet.deploy.prototxt',
        'size': (227, 227),  # cf. bvlc_alexnet.deploy.prototxt
        'mean': (104, 117, 123),  # approximation, should be done by feature
        'crop': True,
        'swap_rb': False
       },
    2: {
        'name': 'GoogleNet',
        'folder': '../data/models_caffe/classification/',
        'model_file': 'bvlc_googlenet.caffemodel',
        'config_file': 'bvlc_googlenet.deploy.prototxt',
        'size': (224, 224),  # cf. bvlc_googlenet.deploy.prototxt
        'mean': (104, 117, 123),  # cf. train_val.prototxt
        'crop': True,  # ? False is used here : https://docs.opencv.org/3.4/d5/de7/tutorial_dnn_googlenet.html
        'swap_rb': False
       },
    3: {
        'name': 'VGG16',
        'folder': '../data/models_caffe/classification/',
        'model_file': 'VGG_ILSVRC_16_layers.caffemodel',
        'config_file': 'VGG_ILSVRC_16_layers_deploy.prototxt',
        'size': (224, 224),  # cf. VGG_ILSVRC_16_layers_deploy.prototxt
        'mean': (103.939, 116.779, 123.68),  # cf. https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
        'crop': True,
        'swap_rb': False
       }
}

#------------------------------------------------------------------------------

def browse_and_classify():

    model_data = MODELS[OPTION_MODEL]
    model_data['n_top'] = N_TOP
    model_data['labels'] = util_ilsvrc.read_labels()

    print(f'Image classification with CNN ({model_data["name"]})')
    try:
        model = util_ocv_dnn.Model(model_data)
    except FileNotFoundError:
        return

    separ = '\n' + '-' * 80 + '\n\n'
    for root, dirs, files in os.walk(IMG_FOLDER):
        for file in files:
            print(f'{separ}{file}\n')
            img = cv.imread(os.path.join(root, file))
            labels = model.predict(img)
            for proba, label in labels:
                print(f'{100*proba:4.0f} | {label}')

#------------------------------------------------------------------------------

if __name__ == '__main__':

    browse_and_classify()
