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
