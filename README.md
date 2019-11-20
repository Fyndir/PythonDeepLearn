# Le deep learning en python

## Remerciement

 Tout le code à ete fourni par  Rene Ebel
 
 rene.ebel@orange.com

## Prérequis : 

Placer vous dans le dossier **librairies** et passer les commandes suivantes, dans l'ordre indiqué :

```bash 
sudo pip install numpy
sudo pip install six
sudo pip install Pillow
sudo pip install opencv_python
sudo pip install torch
sudo pip install torchvision
sudo pip install matplotlib
sudo pip installscipy

cd data/models_caffe/classification
wget http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
wget http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel

cd ../face_analysis
wget https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel
wet https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel

cd ../face_detection
wget https://github.com/sr6033/face-detection-with-OpenCV-and-DNN/raw/master/res10_300x300_ssd_iter_140000.caffemodel

cd ../../models_pytorch/classification
wget https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
wget https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth
wget https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth

```


## D1_Webcam : 

le programme suivant permet d'accédé au flux video renvoyer par le webcam du pc 

```py

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
```

## D2_image_stitcher

le programme permet de fusionner trois image en une photo panoramique avec la librairie **openCv**

```py
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
```

## D3_classify_pth

Ce programme permet de classer grace a un des modeles de reseaux de neurone situer dans le dossier **../data/models_pytorch/classification/** toute les photos contenu dans le dossier **../data/photo_1** grace à la librairie **PyTorch**

```py
"""Image classification with pretrained CNNs (PyTorch).

Main references:
    https://pytorch.org/hub/pytorch_vision_alexnet
    https://pytorch.org/hub/facebookresearch_WSL-Images_resnext
"""

import os

import PIL
import torch
import torchvision

import util_ilsvrc

#------------------------------------------------------------------------------
 
IMG_FOLDER = '../data/photo_1'

# 1: AlexNet
# 2: ResNeXt-101 32x8d
# 3: ResNeXt-101 32x48d
OPTION_MODEL = 1

N_TOP = 5

#------------------------------------------------------------------------------

MODEL_FOLDER = '../data/models_pytorch/classification/'

MODELS = {
    1: {
        'name': 'AlexNet',
        'file': 'alexnet-owt-4df8aa71.pth',
        'class': torchvision.models.AlexNet,
        'kwargs': {}
       },
    2: {
        'name': 'ResNeXt-101 32x8d',
        'file': 'ig_resnext101_32x8-c38310e5.pth',
        'class': torchvision.models.ResNet,
        'kwargs': {
            'block': torchvision.models.resnet.Bottleneck,
            'layers': [3, 4, 23, 3],
            'groups': 32,
            'width_per_group': 8}
       },
    3: {
        'name': 'ResNeXt-101 32x48d',
        'file': 'ig_resnext101_32x48-3e41cc8a.pth',
        'class': torchvision.models.ResNet,
        'kwargs': {
            'block': torchvision.models.resnet.Bottleneck,
            'layers': [3, 4, 23, 3],
            'groups': 32,
            'width_per_group': 48}
       }
}

#------------------------------------------------------------------------------

class Model():

    def __init__(self, data):

        self.model = data['class'](**data['kwargs'])
        name = data['folder'] + data['file']
        try:
            state_dict = torch.load(name)
        except FileNotFoundError:
            print(f'\nERROR: model data file not found ({name})')
            print('\nThat file is to be manually downloaded first.')
            raise
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.labels = data['labels']

    def predict(self, img, no_top=1):

        tt = torchvision.transforms
        preprocess = tt.Compose([
            tt.Resize(256),
            tt.CenterCrop(224),
            tt.ToTensor(),
            tt.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_batch)
        scores = torch.nn.functional.softmax(output[0], dim=0)
        scores = torch.topk(scores, k=no_top)
        return [(value.item(), self.labels[index])
            for value, index in zip(*scores)]

#------------------------------------------------------------------------------

def browse_and_classify():

    model_data = MODELS[OPTION_MODEL]
    model_data['folder'] = MODEL_FOLDER
    model_data['labels'] = util_ilsvrc.read_labels()
    print(f'Image classification with CNN ({model_data["name"]})')
    try:
        model = Model(model_data)
    except FileNotFoundError:
        return

    for root, dirs, files in os.walk(IMG_FOLDER):
        for file in files:
            print(f'\n{file}\n')
            img = PIL.Image.open(os.path.join(root, file))
            labels = model.predict(img, N_TOP)
            for proba, label in labels:
                print(f'{100*proba:4.0f} | {label}')

#------------------------------------------------------------------------------

if __name__ == '__main__':

    browse_and_classify()

```

## D4_classify_ocv

Ce programme permet de classer grace a un des modeles de reseaux de neurone situer dans le dossier **../data/models_pytorch/classification/** toute les photos contenu dans le dossier **../data/photo_1** grace à la librairie **openCV**


```py
"""Image classification with pretrained CNNs (OpenCV, Caffe models).
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

```

## D5_convolutions

Application en traitement d'images

```py
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

img = convert_to_gray(plt.imread('../data/zep.jpg')).astype(np.float32)
imshow(img, 141, K1, 'identity')
imshow(img, 142, K2, 'box blur')
imshow(img, 143, K3, 'sharpen')
imshow(img, 144, K4, 'edge detection', True)
plt.subplots_adjust(left=0.01, right=0.99, wspace=0.05)
plt.show()

```

## D6_alexnet_visu

Visualisation des 96 noyaux (de taille 11×11×3) appris par la première couche de convolution du model Alexnet.

```py
"""Alexnet: visualization of the first convolution layer.
"""

import cv2 as cv
import numpy as np
import torch

#------------------------------------------------------------------------------

MODEL_FILE = '../data/models_pytorch/classification/alexnet-owt-4df8aa71.pth'

def read_model_data():

    name = MODEL_FILE
    try:
        state_dict = torch.load(name)
    except FileNotFoundError:
        print(f'\nERROR: model data file not found ({name})')
        raise
    return state_dict

#------------------------------------------------------------------------------

def show(data):

    nd, _, s1, s1 = data.shape
    MARGIN = 10
    SCALE = 4
    s2 = s1 * SCALE
    step = s2 + MARGIN
    NC = 8
    wc = step*NC + MARGIN
    hc = step*NC + MARGIN
    canvas = np.full((hc+1, wc+1, 3), 255, dtype=np.uint8)
    id = 0
    ic = 0
    x = MARGIN
    y = MARGIN
    while True:
        a = data[id]
        a = np.transpose(a, (1,2,0))
        a -= np.min(a)
        m = 255/np.max(a)
        a = np.round(m*a).astype(np.uint8)
        a = cv.resize(a, None, fx=SCALE, fy=SCALE, interpolation=cv.INTER_AREA)
        canvas[y:y+s2, x:x+s2, :] = a
        id += 1
        if id == nd:
            break
        ic += 1
        if ic == NC:
            ic = 0
            x = MARGIN
            y += step
        else:
            x += step
    cv.imshow('Alexnet', canvas)
    key = cv.waitKey(0)

#------------------------------------------------------------------------------

if __name__ == '__main__':

    state_dict = read_model_data()
    weight = state_dict['features.0.weight'].detach().numpy()
    show(weight)


```

## d7_alexnet_expe_torch

Utilise le model Alexnet pour deviner le contenu d'une image avec Pytorch

```py
"""Alexnet: experimental implementation with PyTorch (prediction only).
"""

import PIL.Image
import numpy as np
import torch
import torchvision

import util_ilsvrc

#------------------------------------------------------------------------------

IMG_FILE = '../data/photo_1/1_tick.jpg'

N_TOP = 5

MODEL_FILE = '../data/models_pytorch/classification/alexnet-owt-4df8aa71.pth'

#------------------------------------------------------------------------------

def read_model_data(name):

    try:
        state_dict = torch.load(name)
    except FileNotFoundError:
        print(f'\nERROR: model data file not found ({name})')
        raise
    return state_dict

#------------------------------------------------------------------------------

def classify_with_alexnet(img, state_dict, labels, n_top):

    def get_data(*keys):
        return tuple(state_dict[key] for key in keys)

    #----- preprocessing

    tt = torchvision.transforms
    preprocess = tt.Compose([
        tt.Resize(256),
        tt.CenterCrop(224),
        tt.ToTensor(),
        tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    x = preprocess(img)
    x = preprocess(img).unsqueeze(0)

    #----- feature extraction

    tf = torch.nn.functional

    weight, bias = get_data('features.0.weight', 'features.0.bias')
    x = tf.conv2d(input=x, weight=weight, bias=bias, stride=4, padding=2)
    x = tf.relu(x)
    x = tf.max_pool2d(x, kernel_size=3, stride=2)

    weight, bias = get_data('features.3.weight', 'features.3.bias')
    x = tf.conv2d(input=x, weight=weight, bias=bias, padding=2)
    x = tf.relu(x)
    x = tf.max_pool2d(x, kernel_size=3, stride=2)

    weight, bias = get_data('features.6.weight', 'features.6.bias')
    x = tf.conv2d(input=x, weight=weight, bias=bias, padding=1)
    x = tf.relu(x)

    weight, bias = get_data('features.8.weight', 'features.8.bias')
    x = tf.conv2d(input=x, weight=weight, bias=bias, padding=1)
    x = tf.relu(x)

    weight, bias = get_data('features.10.weight', 'features.10.bias')
    x = tf.conv2d(input=x, weight=weight, bias=bias, padding=1)
    x = tf.relu(x)
    x = tf.max_pool2d(x, kernel_size=3, stride=2)

    # present in the original code, but not necessary here
    # x = tf.adaptive_avg_pool2d(x, output_size=6)

    x = x.view(x.size(0), 256 * 6 * 6)

    #----- classification

    weight = state_dict['classifier.1.weight']
    bias = state_dict['classifier.1.bias']
    x = tf.linear(x, weight, bias)
    x = tf.relu(x)

    weight = state_dict['classifier.4.weight']
    bias = state_dict['classifier.4.bias']
    x = tf.linear(x, weight, bias)
    x = tf.relu(x)

    weight = state_dict['classifier.6.weight']
    bias = state_dict['classifier.6.bias']
    x = tf.linear(x, weight, bias)

    #----- scores and labels

    scores = tf.softmax(x[0], dim=0)
    scores = torch.topk(scores, k=n_top)
    result = [(value.item(), labels[index]) for value, index in zip(*scores)]
    return result

#------------------------------------------------------------------------------

if __name__ == '__main__':

    labels = util_ilsvrc.read_labels()
    state_dict = read_model_data(MODEL_FILE)
    img = PIL.Image.open(IMG_FILE)
    result = classify_with_alexnet(img, state_dict, labels, N_TOP)
    for proba, label in result:
        print(f'{100*proba:4.0f} | {label}')
```

## D8_alexnet_expe_np

Utilise le model Alexnet pour deviner le contenu d'une image avec NumPy

```py
"""Alexnet: experimental implementation with NumPy (prediction only).
"""

import time

import PIL.Image
import numpy as np

from torch import load

import sys
sys.path.append('../common')
import util_ilsvrc

#------------------------------------------------------------------------------

IMG_FILE = '../data/photo_1/1_tick.jpg'

N_TOP = 5

MODEL_FILE = '../data/models_pytorch/classification/alexnet-owt-4df8aa71.pth'

#------------------------------------------------------------------------------

def read_model_data(name):

    try:
        state_dict = load(name)
    except FileNotFoundError:
        print(f'\nERROR: model data file not found ({name})')
        raise

    # convert tensors to ndarrays
    for key in state_dict:
        state_dict[key] = state_dict[key].detach().numpy()

    return state_dict

#------------------------------------------------------------------------------

def resize(img, size):
    """= torchvision.transforms.functional.resize"""

    w, h = img.size
    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)
    return img.resize((ow, oh), PIL.Image.BILINEAR)

#------------------------------------------------------------------------------

def center_crop(img, output_size):
    """= torchvision.transforms.functional.center_crop"""

    w, h = img.size
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return img.crop((j, i, j + tw, i + th))

#------------------------------------------------------------------------------

def normalize(img, mean, std):
    """= torchvision.transforms.functional.normalize"""

    for i in range(3):
       img[i,:,:] = (img[i,:,:]-mean[i])/std[i]
    return img

#------------------------------------------------------------------------------

def correlate_and_add(input, kernel, output, stride, padding):
    """Compute: output = output + correlate(input, kernel)."""

    hi, wi = input.shape
    kh, kw = kernel.shape
    ho = int((hi+2*padding-(kh-1)-1)/stride + 1)
    wo = int((wi+2*padding-(kw-1)-1)/stride + 1)
    input = np.pad(input, pad_width=padding, mode='constant')
    for i in range(wo):
        si = i*stride
        for j in range(ho):
            sj = j*stride
            output[j,i] += np.sum(input[sj:sj+kh, si:si+kw]*kernel)

#------------------------------------------------------------------------------

def conv2d(input, weight, bias, stride=1, padding=0):
    """= torch.nn.functional.conv2d"""

    n, ci, hi, wi = input.shape
    co, ci_w, kh, kw = weight.shape
    if ci != ci_w:
        raise ValueError('incompatible input and weight shapes')
    co_w = bias.shape[0]
    if co != co_w:
        raise ValueError('incompatible weight and bias shapes')
    ho = int((hi+2*padding-(kh-1)-1)/stride + 1)
    wo = int((wi+2*padding-(kw-1)-1)/stride + 1)
    output = np.empty((n, co, ho, wo), dtype=np.float32)
    for ib in range(n):
        for ico in range(co):
            output[ib,ico,:,:].fill(bias[ico])
            for ici in range(ci):
                correlate_and_add(input[ib,ici,:,:], weight[ico,ici,:,:],
                    output[ib,ico,:,:], stride, padding)
    return output

#------------------------------------------------------------------------------

def relu(input):
    """= torch.nn.functional.relu"""

    return np.maximum(input, 0)

#------------------------------------------------------------------------------

def max_pool2d(input, kernel_size, stride):
    """= torch.nn.functional.max_pool2d"""

    n, c, hi, wi = input.shape
    ho = int((hi-(kernel_size-1)-1)/stride + 1)
    wo = int((wi-(kernel_size-1)-1)/stride + 1)
    output = np.empty((n, c, ho, wo), dtype=np.float32)
    for ib in range(n):
        for ic in range(c):
            for ih in range(ho):
                h0 = stride * ih
                h1 = stride * ih + kernel_size
                for iw in range(wo):
                    w0 = stride * iw
                    w1 = stride * iw + kernel_size
                    output[ib,ic,ih,iw] = np.max(input[ib,ic,h0:h1,w0:w1])
    return output

#------------------------------------------------------------------------------

def linear(input, weight, bias):
    """= torch.nn.functional.linear"""

    return np.matmul(input, weight.T) + bias

#------------------------------------------------------------------------------

def softmax(input):
    """= torch.nn.functional.softmax"""

    e = np.exp(input)
    return e / e.sum()

#------------------------------------------------------------------------------

def topk(input, k):
    """= torch.topk"""

    a = - input
    ind = np.argpartition(a, k-1)[:k]
    ind = ind[np.argsort(a[ind])]
    return input[ind], ind

#------------------------------------------------------------------------------

def classify_with_alexnet(img, state_dict, labels, n_top):

    def get_data(*keys):
        return tuple(state_dict[key] for key in keys)

    #----- preprocessing

    img = resize(img, 256)
    img = center_crop(img, (224, 224))
    x = np.array(img)
    x = np.array((x[:,:,0], x[:,:,1], x[:,:,2]), np.float32) / 255
    x = normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    x = x[np.newaxis]

    #----- feature extraction

    print('conv 1')
    weight, bias = get_data('features.0.weight', 'features.0.bias')
    x = conv2d(x, weight, bias, stride=4, padding=2)   
    x = relu(x)
    x = max_pool2d(x, kernel_size=3, stride=2)

    print('conv 2')
    weight, bias = get_data('features.3.weight', 'features.3.bias')
    x = conv2d(x, weight, bias, padding=2)
    x = relu(x)
    x = max_pool2d(x, kernel_size=3, stride=2)

    print('conv 3')
    weight, bias = get_data('features.6.weight', 'features.6.bias')
    x = conv2d(x, weight, bias, padding=1)
    x = relu(x)

    print('conv 4')
    weight, bias = get_data('features.8.weight', 'features.8.bias')
    x = conv2d(x, weight, bias, padding=1)
    x = relu(x)

    print('conv 5')
    weight, bias = get_data('features.10.weight', 'features.10.bias')
    x = conv2d(x, weight, bias, padding=1)
    x = relu(x)
    x = max_pool2d(x, kernel_size=3, stride=2)
    x = x.flatten()

    #----- classification

    print('class 1')
    weight, bias = get_data('classifier.1.weight', 'classifier.1.bias')
    x = linear(x, weight, bias)
    x = relu(x)

    print('class 2')
    weight, bias = get_data('classifier.4.weight', 'classifier.4.bias')
    x = linear(x, weight, bias)
    x = relu(x)

    print('class 3\n')
    weight, bias = get_data('classifier.6.weight', 'classifier.6.bias')
    x = linear(x, weight, bias)

    #----- scores and labels

    scores = softmax(x)
    scores = topk(scores, n_top)
    result = [(value, labels[index]) for value, index in zip(*scores)]
    return result

#------------------------------------------------------------------------------

if __name__ == '__main__':

    print('Processing, please wait...\n')
    labels = util_ilsvrc.read_labels()
    state_dict = read_model_data(MODEL_FILE)
    img = PIL.Image.open(IMG_FILE)
    start = time.process_time()
    result = classify_with_alexnet(img, state_dict, labels, N_TOP)
    end = time.process_time()
    print(f'time: {round(end-start)} s\n')
    for proba, label in result:
        print(f'{100*proba:4.0f} | {label}')


```

## D9_face_analysis_ocv

Utilise un reseau de neurone pour localiser les visages et definir le sexe et l'age

```py
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

```

## D10_perturb

Applique un filtre au photo situer dans le repertoire **data/photo_1/** pour les rendre inutilisable par les reseaux de neurones de google

```py
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
```

## D11_patch

Insere l'image **patch_vgg16 dans** toute les image du dossier **../data/photo_1**

```py
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
```

