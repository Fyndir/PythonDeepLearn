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
