from typing import List
from torch import Tensor
from multi_style_transfer.helper_functions import *
from multi_style_transfer.StyleMixer import StyleMixer


# WARNING: Do not import any other libraries or files

def normalize(img, mean, std):
    """ Z-normalizes an image tensor.

    # Parameters:
        @img, torch.tensor of size (b, c, h, w)
        @mean, torch.tensor of size (c)
        @std, torch.tensor of size (c)

    # Returns the normalized image
    """
    # TODO: 1. Implement normalization doing channel-wise z-score normalization.
    # Do not use for-loops, make use of Pytorch vectorized operations.

    # channel-wise normalization
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    img = (img - mean) / std

    return img


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
    # TODO: 2. Implement the content loss given the input feature volume and the
    # content feature volume. Note that:
    # - Only the layers given in content_layers should be used for calculating this loss.
    # - Normalize the loss by the number of layers.

    loss = 0.0

    for layer in content_layers:
        input_feat_activation = input_features[layer]
        content_feat_activation = content_features[layer].detach()

        # Compute Mean Squared Error between the input feature activations and the content feature activations
        mse_loss = torch.nn.MSELoss(reduction='mean')

        mse_content = mse_loss(input_feat_activation, content_feat_activation)

        loss = loss + mse_content

    # Normalize loss by number of layers
    loss = loss / len(content_layers)

    return loss


def gram_matrix(x):
    """ Calculates the gram matrix for a given feature matrix.

    # NOTE: Normalize by number of number of dimensions of the feature matrix.

    # Parameters:
        @x, torch.tensor of size (b, c, h, w)

    # Returns the gram matrix
    """
    # TODO: 3.2 Implement the calculation of the normalized gram matrix.
    # Do not use for-loops, make use of Pytorch functionalities.

    # Get normalization factors
    b = x.shape[0]
    nc = x.shape[1]
    nh = x.shape[2]
    nw = x.shape[3]

    # View feature matrix as matrix with b batches, c rows and h * w columns
    x_mod = x.view(b, nc, nh * nw)

    # Obtain & normalize Gram matrix
    x = torch.bmm(x_mod, x_mod.transpose(1, 2)) / (b * nc * nh * nw)

    return x


def style_loss(input_features, style_features:List[Tensor], style_layers, style_img_weight):
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
        @style_img_weight, the weight of the style feature on the right (for multi-style transfer)

    # Returns the style loss, a torch.tensor of size (1)
    """
    # TODO: 3.1 Implement the style loss given the input feature volume and the
    # style feature volume. Note that:
    # - Only the layers given in style_layers should be used for calculating this loss.
    # - Normalize the loss by the number of layers.
    # - Implement the gram_matrix function.

    loss = 0.0

    for layer in style_layers:

        # normalized Gram matrix for input image
        input_feat_activation = gram_matrix(input_features[layer])

        if len(style_features) == 1:
            # normalized Gram matrix for style image
            style_feat_activation = gram_matrix(style_features[0][layer])
        else:
            # Mix all of the style features for multi-style transfer
            style_feats_per_layer = [feats[layer] for feats in style_features]
            style_mixer = StyleMixer(style_feats_per_layer, style_img_weight)
            style_feat_activation = style_mixer.mix()
            style_feat_activation = gram_matrix(style_feat_activation)

        # Compute Mean Squared Error between the input gram matrix and the style gram matrix
        mse_loss = torch.nn.MSELoss(reduction='mean')

        mse_content = mse_loss(input_feat_activation, style_feat_activation)

        loss = loss + mse_content

    loss = loss / len(style_layers)

    return loss


def total_variation_loss(y):
    """ Calculates the total variation across the spatial dimensions.

    # Parameters:
        @x, torch.tensor of size (b, c, h, w)

    # Returns the total variation, a torch.tensor of size (1)
    """
    # TODO: 4. Implement the total variation loss. Normalize by tensor dimension sizes
    # Do not use for-loops, make use of Pytorch vectorized operations.

    # Normalize by c * K * J
    norm_factor = y.shape[1] * y.shape[2] * y.shape[3]

    # Obtain I[k + 1, j] - I[k, j]
    k_diff = y[:, :, 1:, :] - y[:, :, :-1, :]

    # Obtain I[k, j + 1] - I[k, j]
    j_diff = y[:, :, :, 1:] - y[:, :, :, :-1]

    # Obtain |I[k + 1, j] - I[k, j]| + |I[k, j + 1] - I[k, j]|
    abs_diff = torch.sum(torch.abs(k_diff)) + torch.sum(torch.abs(j_diff))

    tv_loss = abs_diff / norm_factor

    return tv_loss


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
    # TODO: 5. Calculate the gradient images based on the sobel kernel
    # Do not use for-loops, make use of Pytorch vectorized operations.

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
    # TODO: 6. Calculate the edge loss
    # Do not use for-loops, make use of Pytorch vectorized operations.

    mse_loss = torch.nn.MSELoss(reduction='mean')

    # Compute edge loss on both x and y-axes
    edge_loss_dx = mse_loss(img1[:, 0, :, :], img2[:, 0, :, :])
    edge_loss_dy = mse_loss(img1[:, 1, :, :], img2[:, 1, :, :])

    return (edge_loss_dx + edge_loss_dy) / 2
