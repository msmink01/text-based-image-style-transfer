import copy
from typing import List
from torch import Tensor
import torch.nn.functional as F
import numpy as np

class StyleMixer():
    def __init__(self, style_feat_list:List[Tensor], style_feat_weight:float):
        """"
        Fuses multiple style images in order to obtain a style-mixed output.
        @param style_feat_list: Contains all style image features to fuse. The image features are stored in a tensor of
            shape [B, C, H, W]
        @param style_feat_weight: List of weights for each style image. Images with a higher weight will have a higher
        style contribution to the final style.
        """

        style1 = style_feat_list[0].detach()
        style2 = style_feat_list[1].detach()

        self.style_feat_list = [copy.deepcopy(style1), copy.deepcopy(style2)]

        # Ensure that the weights sum to 1
        self.style_feat_weights = [1 - style_feat_weight, style_feat_weight]

    def mix(self):
        """"
        Takes two style images and their weights, and returns a tensor with the combined features of the two.
        @return mixed style image features.
        """
        # Result size: [1, C, H, W]
        midpoint_shape = tuple(int(x) for x in np.array(self.style_feat_list[0].shape[2:]) +
                               np.array(self.style_feat_list[1].shape[2:]) // 2)

        self.style_feat_list[0] = F.interpolate(self.style_feat_list[0], size=midpoint_shape, mode='bilinear', align_corners=True)
        self.style_feat_list[1] = F.interpolate(self.style_feat_list[1], size=midpoint_shape, mode='bilinear', align_corners=True)
        
        result = self.style_feat_weights[0] * self.style_feat_list[0] + self.style_feat_weights[1] * self.style_feat_list[1]
        return result
