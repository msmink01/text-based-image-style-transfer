from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np

class GhiasiImgTransformer(nn.Module):
    def __init__(self):
        """
        Create a GhiasiImgTransformer object

        Inpired by: https://doi.org/10.48550/arXiv.1705.06830
        """
        super(GhiasiImgTransformer,self).__init__()

        self.layers = nn.ModuleList([
            ConvInRelu(3,32,9,stride=1), # Encoder layers
            ConvInRelu(32,64,3,stride=2),
            ConvInRelu(64,128,3,stride=2),
            ResidualBlock(128), # Transforming layers
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            UpsampleConvInRelu(128,64,3,upsample=2), # Decoder layers
            UpsampleConvInRelu(64,32,3,upsample=2),
            UpsampleConvInRelu(32,3,9,upsample=None,activation=None)
        ])

        # Load in weights
        checkpoint_ghiasi = torch.load('text/subnetworks/checkpoints/image_transformer.pth', map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint_ghiasi['state_dict_ghiasi'])

        self.n_params = sum([layer.n_params for layer in self.layers])
    
    def forward(self, x, styles):
        """
        Forward pass of this GhiasiImgTransformer object.

        Args:
            x (Tensor): input tensor of shape B x 3 x H x W or 3 x H x W
            styles (Tensor): style embedding tensor of shape B x 100

        Returns:
            output (Tensor): output tensor very similar in shape to original x with some slight changes due to padding rounding
        """
        # x: B x 3 x H x W or 3 x H x W
        # styles: B x 100 batch of style embeddings
        for i, layer in enumerate(self.layers):
            if i < 3:
                # first three layers do not perform renormalization (see style_normalization_activations in the original source: https://github.com/tensorflow/magenta/blob/master/magenta/models/arbitrary_image_stylization/nza_model.py)
                x = layer(x)
            else:
                x = layer(x, styles)
        
        return torch.sigmoid(x)
    
class ConvInRelu(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride=1):
        """
        Create a ConvInRelu block. 
        
        This block first applies reflection padding, then a convolutional layer with instance normalization, followed by a ReLU activation.

        Args:
            channels_in (int): number of input channels to conv2d layer
            channels_out (int): number of output channels to conv2d layer
            kernel_size (int): kernel size of the conv2d layer
            stride (int): stride of the conv2d layer
        """
        super(ConvInRelu,self).__init__()
        self.n_params = 0
        self.channels = channels_out
        self.reflection_pad = nn.ReflectionPad2d(int(np.floor(kernel_size/2)))
        self.conv = nn.Conv2d(channels_in,channels_out,kernel_size,stride,padding=0)
        self.instancenorm = nn.InstanceNorm2d(channels_out)
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x):
        """
        Forward pass of a ConvInRelu block.
        
        First applies reflection padding, then a convolutional layer with instance normalization, followed by a ReLU activation.

        Args:
            x (Tensor): input tensor with shape (B x channels_in x H x W)

        Returns:
            x (Tensor): output tensor with shape (B x channels_out x H' x W')
        """
        x = self.reflection_pad(x)
        x = self.conv(x)
        x = self.instancenorm(x)
        x = self.relu(x)
        return x


class UpsampleConvInRelu(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, upsample, stride=1, activation=nn.ReLU):
        """
        Create a UpsampleConvInRelu block. 
        
        Similar to the ConvInRelu block but has an upsampling sayer.
        This block first applies upsampling, reflection padding, then a convolutional layer with instance normalization, followed by adding two fully connected layers of the style, finally an activation.

        Args:
            channels_in (int): number of input channels to conv2d layer
            channels_out (int): number of output channels to conv2d layer
            kernel_size (int): kernel size of the conv2d layer
            upsample (int): the factor by which input tensor is unsampled (None if no upsampling)
            stride (int): stride of the conv2d layer
            activation (nn.Module): activation to use
        """
        super(UpsampleConvInRelu, self).__init__()
        self.n_params = channels_out * 2
        self.upsample = upsample
        self.channels = channels_out

        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, stride)
        self.instancenorm = nn.InstanceNorm2d(channels_out)
        self.fc_beta = nn.Linear(100,channels_out)
        self.fc_gamma = nn.Linear(100,channels_out)
        if activation:
            self.activation = activation(inplace=False)
        else:
            self.activation = None
        
    def forward(self, x, style):
        """
        Forward pass of a UpsampleConvInRelu block.
        
        First applies upsampling, reflection padding, then a convolutional layer with instance normalization, followed by adding two fully connected layers of the style, finally an activation.

        Args:
            x (Tensor): input tensor with shape (B x channels_in x H x W)
            style (Tensor): style embedding tensor with shape (B x 100)

        Returns:
            x (Tensor): output tensor with shape (B x channels_out x H' x W')
        """
        # Get style learned scale (gamma) and shift (beta) parameters
        beta = self.fc_beta(style).unsqueeze(2).unsqueeze(3) # B x C_out x 1 x 1
        gamma = self.fc_gamma(style).unsqueeze(2).unsqueeze(3) # B x C_out x 1 x 1

        # Transform input tensor
        if self.upsample:
            x = self.upsample_layer(x)
        x = self.reflection_pad(x)
        x = self.conv(x)
        x = self.instancenorm(x)
        x = gamma * x # apply scale
        x += beta # apply shift
        if self.activation:
            x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        """
        Create a ResidualBlock.

        Inspired by:  https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
        
        Consists of two convolutional layers (each with kernel size=3), each followed by instance normalization and ReLU activation. Also includes
            style based modulation of the output using the style embeddings. Contains skip connection between input and output.

        Args:
            channels (int): number of input and output channels
        """
        super(ResidualBlock,self).__init__()
        self.n_params = channels * 4
        self.channels = channels

        self.reflection_pad = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(channels,channels,3,stride=1,padding=0)
        self.instancenorm = nn.InstanceNorm2d(channels)
        self.fc_beta1 = nn.Linear(100,channels)
        self.fc_gamma1 = nn.Linear(100,channels)
        self.fc_beta2 = nn.Linear(100,channels)
        self.fc_gamma2 = nn.Linear(100,channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(channels,channels,3,stride=1,padding=0)
        
    def forward(self, x, style):
        """
        Forward pass of a ResidualBlock block.
        
        Consists of two convolutional layers (each with kernel size=3), each followed by instance normalization and ReLU activation. Also includes
            style based modulation of the output using the style embeddings. Contains skip connection between input and output.

        Args:
            x (Tensor): input tensor with shape (B x channels x H x W)
            style (Tensor): style embedding tensor with shape (B x 100)

        Returns:
            x (Tensor): output tensor with shape (B x channels x H' x W')
        """
        # Get style learned scale (gamma) and shift (beta) parameters
        beta1 = self.fc_beta1(style).unsqueeze(2).unsqueeze(3) # B x C x 1 x 1
        gamma1 = self.fc_gamma1(style).unsqueeze(2).unsqueeze(3) # B x C x 1 x 1
        beta2 = self.fc_beta2(style).unsqueeze(2).unsqueeze(3) # B x C x 1 x 1
        gamma2 = self.fc_gamma2(style).unsqueeze(2).unsqueeze(3) # B x C x 1 x 1

        # Transform input tensor
        y = self.reflection_pad(x)
        y = self.conv1(y)
        y = self.instancenorm(y)
        y = gamma1 * y # scale
        y += beta1 # shift
        y = self.relu(y)
        y = self.reflection_pad(y)
        y = self.conv2(y)
        y = self.instancenorm(y)
        y = gamma2 * y # scale
        y += beta2 # shift
        return x + y # skip connection