import cv2
import numpy as np
from PIL import Image
from sklearn.neighbors import KDTree

from components.pixel_art.util import edge_detection, overlay_edges


class PixelArt:
    """
    Converts images to pixel art with configurable pixelation, color palettes, and edge detection.

    Attributes:
        colour_palette: Numpy array of RGB colors for quantization
        palette_tree: KDTree structure for efficient palette lookups
    """
    def __init__(self):
        self.colour_palette = None
        self.palette_tree = None

    def process(self, image, pixel_size=0.3, colour_palette=None, interpolate=False, edge_detect=False, edge_threshold=50):
        """
        Main processing pipeline for pixel art conversion.

        Args:
            image: Input image (NumPy array/PIL Image)
            pixel_size: Image scaling factor (0.0-1.0) for pixel density
            colour_palette: Optional NumPy array (Nx3 RGB) for color quantization
            interpolate: Whether to blend colors in the palette
            edge_detect: Toggle edge overlay
            edge_threshold: Sensitivity for edge detection (0-255)

        Returns:
            PIL.Image: Pixelated output with optional edges
        """

        if not isinstance(image, np.ndarray):
            image = np.asarray(image)

        if colour_palette is not None:
            self.colour_palette = np.asarray(colour_palette.display_palette((1, 256), interpolate=interpolate)).reshape(
                (-1, 3))
            self.palette_tree = KDTree(self.colour_palette,
                                       metric="l2")

        img = image.copy()
        if colour_palette is not None:
            img = self._convert_palette(image)

        if pixel_size <= 0:
            pixel_size = 0.0001
        img, small_img = self._pixelate(img, pixel_size)

        if edge_detect:
            edges = edge_detection(small_img, edge_threshold)
            edges = cv2.resize(edges, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            img = overlay_edges(img, edges)

        return Image.fromarray(img.astype(np.uint8))

    def _pixelate(self, image, pixel_size):
        """
        Creates pixelation effect through dual-phase resizing.

        Args:
            image: Input image array
            pixel_size: Scaling factor for downsampling

        Returns:
            tuple: (Full-size pixelated image, small intermediate image)
        """
        new_size = (int(image.shape[1] * pixel_size), int(image.shape[0] * pixel_size))
        small_img = cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)
        return cv2.resize(small_img, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST), small_img

    def _convert_palette(self, img):
        """
        Maps image colors to nearest palette entries using KDTree search.

        Args:
            img: Input image array

        Returns:
            Numpy array: Color-quantized image
        """
        height, width, _ = img.shape
        img_reshaped = img.reshape((-1, 3))
        _, indices = self.palette_tree.query(img_reshaped)
        return self.colour_palette[indices].reshape((height, width, 3))
