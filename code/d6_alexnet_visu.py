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

