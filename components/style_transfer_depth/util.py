import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


def mask_image_depth(image1, depth, thresholds):
    """
    Masks input image based on normalized depth map thresholds (Original implementation).

    Args:
        image1: Input image (PIL.Image or np.array)
        depth: Depth map (single-channel np.array)
        thresholds: Tuple of (min, max) depth values [0-1 range]

    Returns:
        PIL.Image: Masked image where depth outside thresholds is zeroed
    Raises:
        ValueError: If depth map is multi-channel
    """
    image1 = np.asarray(image1)

    if len(depth.shape) > 2:
        raise ValueError("The depth map (image2) must be a single-channel image.")

    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))

    min_threshold, max_threshold = thresholds
    mask = (depth >= min_threshold) & (depth <= max_threshold)
    masked_image_array = np.copy(image1)
    masked_image_array[~mask] = 0

    return Image.fromarray(masked_image_array)


def create_bins(n):
    """
    Creates equal-width depth bins for MIP processing (Original implementation).

    Args:
        n: Number of depth bins

    Returns:
        list: List of [min, max] tuples defining bin boundaries
    """
    bin_edges = np.linspace(0, 1, n + 1)
    bins = [[bin_edges[i], bin_edges[i + 1]] for i in range(n)]
    return bins


def generate_mip_layers(image, depth, n):
    """
    Generates masked images based on depth bins.

    Args:
        image (PIL.Image or np.array): The input image.
        depth (np.array): The depth map.
        n (int): The number of depth bins.

    Returns:
        list: List of masked images corresponding to depth bins.
    """
    bins = create_bins(n)
    return [mask_image_depth(image, depth, b) for b in bins]


def reconstruct_mip_image(stylized_images, depth, n):
    """
    Reconstructs the final depth-stylized image using depth masks.

    Args:
        stylized_images (list): List of stylized images.
        depth (np.array): The depth map.
        n (int): The number of depth bins.

    Returns:
        PIL.Image: The final depth-aware stylized image.
    """
    bins = create_bins(n)
    final_images = [mask_image_depth(stylized_images[i], depth, bins[i]) for i in range(n)]
    mip = np.zeros((stylized_images[0].size[1], stylized_images[0].size[0], 3), dtype=np.uint8)
    for img in final_images:
        mip += np.asarray(img)
    return Image.fromarray(mip)


# after here it a copy for the code from assignment3/Part1/helper_functions.py

def image_loader(img, device='cpu'):
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    return img.to(device, torch.float)


def save_image(tensor):
    img = tensor.cpu().clone()
    img = img.squeeze(0)
    img = transforms.ToPILImage()(img)
    return img


class Vgg19(torch.nn.Module):
    """
    VGG19 feature extractor for style transfer (From Assignment 3).
    """
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


def normalize(img, mean, std):
    """ Z-normalizes an image tensor.

    # Parameters:
        @img, torch.tensor of size (b, c, h, w)
        @mean, torch.tensor of size (c)
        @std, torch.tensor of size (c)

    # Returns the normalized image
    """
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    normalized_img = (img - mean) / std
    return normalized_img


def content_loss(input_features, content_features, content_layers):
    """ Calculates the content loss as in Gatys et al. 2016.

    # Parameters:
        @input_features, VGG features of the image to be optimized. It is a
            dictionary containing the layer names as keys and the corresponding
            features volumes as values.
        @content_features, VGG features of the content image. It is a dictionary
            containing the layer names as keys and the corresponding features
            volumes as values.
        @content_layers, a list containing which layers to consider for calculating
            the content loss.

    # Returns the content loss, a torch.tensor of size (1)
    """

    content_loss_internal = 0.0

    for layer in content_layers:
        input_feature = input_features[layer]
        content_feature = content_features[layer]
        loss = torch.nn.functional.mse_loss(input_feature, content_feature)
        content_loss_internal += loss
    content_loss_internal = content_loss_internal / len(content_layers)

    return content_loss_internal


def gram_matrix(x):
    """ Calculates the gram matrix for a given feature matrix.

    # NOTE: Normalize by number of dimensions of the feature matrix.

    # Parameters:
        @x, torch.tensor of size (b, c, h, w)

    # Returns the gram matrix
    """
    b, c, h, w = x.size()
    features = x.view(b, c, h * w)
    gram_matrix = torch.bmm(features, features.transpose(1, 2))
    gram_matrix = gram_matrix / (b* c * h * w)

    return gram_matrix


def style_loss(input_features, style_features, style_layers):
    """ Calculates the style loss as in Gatys et al. 2016.

    # Parameters:
        @input_features, VGG features of the image to be optimized. It is a
            dictionary containing the layer names as keys and the corresponding
            features volumes as values.
        @style_features, VGG features of the style image. It is a dictionary
            containing the layer names as keys and the corresponding features
            volumes as values.
        @style_layers, a list containing which layers to consider for calculating
            the style loss.

    # Returns the style loss, a torch.tensor of size (1)
    """

    style_loss = 0.0
    for layer in style_layers:
        input_gram = gram_matrix(input_features[layer])
        style_gram = gram_matrix(style_features[layer])

        loss = torch.nn.functional.mse_loss(input_gram, style_gram)
        style_loss += loss
    style_loss = style_loss / len(style_layers)

    return style_loss


def total_variation_loss(x):
    """ Calculates the total variation across the spatial dimensions.

    # Parameters:
        @x, torch.tensor of size (b, c, h, w)

    # Returns the total variation, a torch.tensor of size (1)
    """

    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    total_variation = torch.sum(torch.abs(tv_h)) + torch.sum(torch.abs(tv_w))
    normalization_factor = x.size(1) * x.size(2) * x.size(3)
    total_variation_loss = total_variation / normalization_factor

    return total_variation_loss


def get_gradient_imgs(img):
    """ Calculates the gradient images based on the sobel kernel.

    # NOTE:
      1. The gradient image along the x-dimension should be at first position,
         i.e. at out[:,0,:,:], and the gradient image calulated along the y-dimension
         should be at out[:,1,:,:].
      2. Do not use padding for the convolution.
      3. When defining the Sobel kernel, use the finite element approximation of the gradient and approximate the derivative in x-direction according to:
            df / dx  =  f(x+1,y) - f(x-1,y)   (value of left neighbor pixel is subtracted from the value of the right neighbor pixel)
         and the derivative in y-direction according to:
            df / dy  =  f(x,y+1) - f(x,y-1)   (value of bottom neighbor pixel is subtracted from the value of the top neighbor pixel)

    # Parameters:
        @img grayscale image, tensor of size (1,1,H,W)

    # Returns the gradient images, concatenated along the second dimension.
      Size (1,2,H-2,W-2)
    """

    # df / dx  =  f(x+1,y) - f(x-1,y)
    dx = img[:, :, 1:img.shape[2] - 1, 2:] - img[:, :, 1:img.shape[2] - 1, :img.shape[3] - 2]
    # df / dy  =  f(x,y+1) - f(x,y-1)
    dy = img[:, :, 2:, 1:img.shape[3] - 1] - img[:, :, :img.shape[2] - 2, 1:img.shape[3] - 1]

    return torch.cat((dx, dy), dim=1)

def edge_loss(img1, img2):
    """ Calculates the edge loss based on the mean squared error between the two images.

    # Parameters:
        @img1 (1,2,H,W)
        @img2 (1,2,H,W)

    # Returns the edge loss, a torch.tensor of size (1)
    """
    mse_loss = torch.nn.MSELoss(reduction='mean')

    # Compute edge loss on both x and y-axes
    edge_loss_dx = mse_loss(img1[:, 0, :, :], img2[:, 0, :, :])
    edge_loss_dy = mse_loss(img1[:, 1, :, :], img2[:, 1, :, :])

    return (edge_loss_dx + edge_loss_dy) / 2


def depth_loss(depth_img, depth_target):
    """
    Custom depth preservation loss (Original implementation).
    Maintains depth consistency between original and stylized images.

    Args:
        depth_img: Generated depth map (1,1,H,W tensor)
        depth_target: Original depth map (1,1,H,W tensor)

    Returns:
        torch.Tensor: MSE loss between depth maps
    """
    mse_loss = torch.nn.functional.mse_loss(depth_img, depth_target)
    return mse_loss
