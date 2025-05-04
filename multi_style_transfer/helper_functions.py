import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def image_loader(path, img_size=128, device='cpu'):
    img = Image.open(path)
    img = transforms.Resize((img_size, img_size))(img)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    return img.to(device, torch.float)


def save_image(tensor, title=None, out_folder=None):
    assert out_folder is not None and title is not None
    plt.figure()
    img = tensor.cpu().clone()
    img = img.squeeze(0)
    img = transforms.ToPILImage()(img)
    plt.imshow(img)

    fig_path = os.path.join(out_folder, f'{title}.png')
    # plt.savefig(fig_path, bbox_inches='tight')
    img.save(fig_path)

    print(f'Saved image with title {title} to {out_folder}')
    plt.close()


class Vgg19(torch.nn.Module):
    def __init__(self, content_layers, style_layers, device):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        self.slices = []
        self.layer_names = []
        self._remaining_layers = set(content_layers + style_layers)
        self._conv_names = [
            'conv1_1', 'conv1_2',
            'conv2_1', 'conv2_2',
            'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
            'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
            'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4',
        ]

        i = 0
        model = torch.nn.Sequential()
        for layer in vgg.children():
            new_slice = False
            if isinstance(layer, nn.Conv2d):
                name = self._conv_names[i]
                i += 1

                if name in content_layers or name in style_layers:
                    new_slice = True
                    self.layer_names.append(name)
                    self._remaining_layers.remove(name)

            elif isinstance(layer, nn.ReLU):
                name = 'relu{}'.format(i)
                layer = nn.ReLU(inplace=False)

            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool{}'.format(i)

            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn{}'.format(i)

            model.add_module(name, layer)

            if new_slice:
                self.slices.append(model)
                model = torch.nn.Sequential()

            if len(self._remaining_layers) < 1:
                break

        if len(self._remaining_layers) > 0:
            raise Exception('Not all layers provided in content_layes and/or style_layers exist.')

    def forward(self, x):
        outs = []
        for slice in self.slices:
            x = slice(x)
            outs.append(x.clone())

        out = dict(zip(self.layer_names, outs))
        return out


def to_grayscale(img):
    """ Simplified way of obtaining a grayscale image.

    # Parameters:
        @img: torch.tensor of size [1,3,H,W]

    # Returns the grayscale image of size torch.tensor of size [1,1,H,W]

    """
    return img.mean(dim=1, keepdim=True)
