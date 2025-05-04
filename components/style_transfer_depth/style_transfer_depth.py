from transformers import pipeline
import numpy as np
from PIL import Image
from components.style_transfer_depth.util import generate_mip_layers, reconstruct_mip_image
from components.style_transfer_depth.Style_a3 import StyleA3


class DepthStyle:
    """
    Performs depth-guided style transfer using a depth estimation model and a style transfer model.
    Supports multi-plan image depth-based stylization.

    Attributes:
        depth_pipeline: Depth estimation model pipeline.
        style_model: Style transfer model (StyleA3).
        style_pipeline: Style transfer function from the model.
    """

    def __init__(self, device="cpu"):
        """
        Initializes the DepthStyle class by loading depth estimation and style transfer models.

        Args:
            device (str): Computation device ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        print("Loading Depth models...")
        self.depth_pipeline = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
        print("Loading Style Transfer models...")
        self.style_model = StyleA3(device=device)
        print(f"DepthStyle initialized on {device}")
        self.style_pipeline = self.style_model.style_transfer

    def get_depth_map(self, image):
        """
        Computes the depth map for a given image.

        Args:
            image (PIL.Image or np.array): Input image.

        Returns:
            np.array: Depth map of the input image.
        """
        depth = self.depth_pipeline(image)["depth"]
        return np.asarray(depth)

    def style_transfer(self, image, style, strength=1):
        """
        Applies style transfer to an image with adjustable style strength.

        Args:
            image (PIL.Image or np.array): Input image.
            style (PIL.Image): Style image.
            strength (float): Style influence strength (0-1). Defaults to 1.

        Returns:
            PIL.Image: Stylized image.
        """
        stylized_image = self.style_pipeline(style, image, strength=strength)
        return stylized_image

    def process_mip_layers(self, masked_images, style):
        """
        Applies style transfer to each depth-masked image with decreasing strength.

        Args:
            masked_images (list): List of depth-masked images.
            style (PIL.Image): Style reference image.

        Returns:
            list: List of stylized images.
        """
        return [self.style_transfer(img, style, (1-ind/len(masked_images))) for ind, img in enumerate(masked_images)]

    def style_MIP(self, image, style, n=2):
        """
        Performs multi-plane image depth style transfer.

        Args:
            image (PIL.Image or np.array): Input image.
            style (PIL.Image): Style image.
            n (int): Number of depth layers. Defaults to 2.

        Returns:
            tuple: (final stylized image, list of stylized depth-layered images) (list of stylized images for debugging)
        """
        depth = self.get_depth_map(image)
        masked_images = generate_mip_layers(image, depth, n)
        stylized_images = self.process_mip_layers(masked_images, style)
        final_image = reconstruct_mip_image(stylized_images, depth, n)
        return final_image, stylized_images

    def style_Dept(self, image, style):
        """
        Performs depth-aware style transfer.

        Args:
            image (PIL.Image or np.array): Input image.
            style (PIL.Image): Style image.

        Returns:
            PIL.Image: Final stylized image.
        """

        stylized_image = self.style_pipeline(style, image, depth=True)
        return stylized_image

    def depth_split(self, image, n=2):
        """
        Splits an image into depth layers.

        Args:
            image (PIL.Image or np.array): Input image.
            n (int): Number of depth layers. Defaults to 2.

        Returns:
            list: List of depth-masked images.
        """
        depth = self.get_depth_map(image)
        return generate_mip_layers(image, np.asarray(depth),  n)