import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

class ColorPaletteTransfer():
    """"
    Applies the color palette of one image to another image. It is important to note that the two
    images should be compatible and depending on the source and target images the results may differ.
    Implementation taken from Reinhard et al. (2001)
    """

    def __init__(self):
        self.rgb_to_lms_transform = torch.FloatTensor([[0.3811, 0.5783, 0.0402],
                                                       [0.1967, 0.7244, 0.0782],
                                                       [0.0241, 0.1288, 0.8444]
                                                      ])
        self.lms_to_ruderman_transform = torch.FloatTensor([[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],
                                                            [1/np.sqrt(6), 1/np.sqrt(6), -2/np.sqrt(6)],
                                                            [1/np.sqrt(2), -1/np.sqrt(2), 0]
                                                           ])
        self.ruderman_to_lms_transform = self.lms_to_ruderman_transform.inverse()
        self.lms_to_rgb_transform = self.rgb_to_lms_transform.inverse()

    def color_transfer_pipeline(self, source_img_filepath, target_img_filepath):
        """"
        Takes as input two different images and transfers the colors of the target image to the source image.
        @param source_img_filepath the path to the image to which the color palette is applied
        @param target_img_filepath the path to the image with the color palette to be applied
        @return a copy of the source image with the color palette transferred from the target image
        """

        # Ensure that the input images are in RGB format
        source_img = transforms.ToTensor()(source_img_filepath.convert('RGB'))
        target_img = transforms.ToTensor()(Image.open(target_img_filepath).convert('RGB'))

        # Pre-process the source and target images
        source_img = torch.clamp(source_img, min=1e-6, max=1.0)
        target_img = torch.clamp(target_img, min=1e-6, max=1.0)

        if source_img.shape != target_img.shape:
            target_img = F.interpolate(target_img.unsqueeze(0), source_img.shape[1:], mode='bilinear',
                                       align_corners=True).squeeze(0)
        final_shape = source_img.shape

        # Transform to 2d. The result will be of shape [H * W, 3], where each row corresponds to the RGB values of a pixel
        source_2d = self.to_2d(source_img)
        target_2d = self.to_2d(target_img)

        output = self.color_transfer(source_2d, target_2d)

        # Return to 3d
        output_3d = self.to_3d(output, final_shape)
        output_3d = torch.clamp(output_3d, min=0.0, max=1.0)

        return transforms.ToPILImage()(output_3d.detach())


    def color_transfer(self, source_img, target_img):
        """"
        Transfers the color palette of the target image to the source image.
        @param source_img the image to which the color palette is applied
        @param target_img the image with the color palette to be applied
        """

        # Convert to Ruderman color space
        source_ruderman = self.rgb_to_ruderman(source_img)
        target_ruderman = self.rgb_to_ruderman(target_img)

        # Avoid division by 0
        source_stds = source_ruderman.std(dim=0, keepdim=True)
        source_stds = torch.where(source_stds < 1e-5, torch.ones_like(source_stds), source_stds)

        target_stds = target_ruderman.std(dim=0, keepdim=True)
        source_means = source_ruderman.mean(dim=0, keepdim=True)
        target_means = target_ruderman.mean(dim=0, keepdim=True)

        # Scale the data points from ONLY THE SOURCE IMAGE
        source_ruderman = source_ruderman - source_means
        final_source = source_ruderman * (target_stds / source_stds)

        # Transfer the mean of the target to the source
        final_img = final_source + target_means

        # Convert back to RGB
        final_img = self.ruderman_to_rgb(final_img)

        return final_img


    def rgb_to_ruderman(self, img:torch.Tensor):
        """"
        Convert an image from the RGB color space to the Ruderman (L alpha beta) color space.
        @param img an RGB image tensor. The image is arranged as [R, G, B], where the red green and blue values are three
        vectors repr. a 1d view of the image
        @return an image in the Ruderman color space.
        """

        # Convert to LMS (long, medium, short) color space
        lms = img @ self.rgb_to_lms_transform.T

        # Eliminate skew by taking the logarithm per (L, M, S) channel
        lms_log = torch.log(lms + 1e-5)

        # Convert to Ruderman color space
        ruderman = lms_log @ self.lms_to_ruderman_transform.T

        return ruderman

    def ruderman_to_rgb(self, img:torch.Tensor):
        """"
        Convert an image from the Ruderman (L alpha beta) color space to the RGB color space.
        @param img an image tensor in the Ruderman space
        @return an RGB image. The image will be arranged as [R, G, B], where the red green and blue values are three
        vectors repr. a 1d view of the image
        """

        # Ruderman to LMS transformation (which is the inverse of LMS to Ruderman transform)
        lms = img @ self.ruderman_to_lms_transform.T

        # Exponentiate to go back to linear space
        lms_linear = torch.exp(lms)

        # Convert from LMS to RGB (inverse of RGB TO LMS)
        rgb = lms_linear @ self.lms_to_rgb_transform.T

        return rgb

    def to_2d(self, img):
        """"
        Takes an image as a tensor of shape [C, H, W] and transforms it into a tensor of shape [H * W, 3]
        """
        result = img.permute(1, 2, 0).reshape(-1, 3)
        return result

    def to_3d(self, img, new_shape):
        """"
        Takes an image as a tensor of shape [H * W, 3] and transforms it into a tensor of shape [3, H, W]
        """
        result = img.reshape(new_shape[1], new_shape[2], 3)
        return result.permute(2, 0, 1)