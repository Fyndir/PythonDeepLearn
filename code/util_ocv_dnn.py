"""Wrapper class for DNN model in OpenCV.

Creation: 2019/08 rene.ebel@orange.com
"""

import cv2 as cv
import numpy as np

#------------------------------------------------------------------------------

def topk(input, k):
    """= torch.topk"""

    a = - input
    ind = np.argpartition(a, k-1)[:k]
    ind = ind[np.argsort(a[ind])]
    return input[ind], ind

#------------------------------------------------------------------------------

class Model():

    def __init__(self, data):
        """
        data keys:
            folder: file folder
            model_file: file containing trained weights
            config_file (optional): network configuration file
            size: size the input image is resized to
            mean: mean values which are subtracted from channels
            for a classifier only:
                n_top: number of tuples (value, category) to return
                labels: tuple of category names
        """
        folder = data['folder'] + '/'
        args = {'model': folder + data['model_file']}
        if 'config_file' in data:
            args['config'] = folder + data['config_file']
        try:
            self.net = cv.dnn.readNet(**args)
        except cv.error:
            print(f'\nERROR: model data files not found in {folder}')
            print('\nThose files are to be manually downloaded first.')
            raise FileNotFoundError
        self.data = data

    def predict(self, img):

        blob = cv.dnn.blobFromImage(img,
            scalefactor = self.data.get('scale_factor', 1.),
            size = self.data['size'],
            mean = self.data['mean'],
            swapRB = self.data['swap_rb'],
            crop = self.data['crop'])
        self.net.setInput(blob)
        pred = self.net.forward()        
        if ('n_top' in self.data) and ('labels' in self.data):
            scores = topk(pred[0], self.data['n_top'])
            return [(value.item(), (self.data['labels'])[index]) for value, index in zip(*scores)]
        else:
            return pred
