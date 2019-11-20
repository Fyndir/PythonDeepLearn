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
