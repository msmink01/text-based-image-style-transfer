import json
from PIL import Image
import numpy as np
from components.pixel_art.util import hex_to_rgb, knn_smoothing


class ColourPalette:
    """
    Manages color palettes for pixel art generation, supporting JSON-based palettes,
    image-derived palettes, and interpolation for gradient effects.

    Attributes:
        palette_list: List of palettes loaded from a JSON file.
        palette: Current active palette as a list of RGB tuples.
    """

    def __init__(self, palette_file="components/pixel_art/100.json", palette_number=None):
        """
        Initializes the ColourPalette object.

        Args:
            palette_file (str): Path to the JSON palette file. Defaults to "100.json". Please Note the formate of the file. = [[#hex code, #hex code, #hex code...], [#hex code, #hex code, #hex code]...]
            palette_number (int, optional): Index of the palette to load. Defaults to None.
        Raises:
            FileNotFoundError: If the palette file is not found.
            IndexError: If the palette number is out of range.
        """
        try:
            with open(palette_file, "r") as f:
                self.palette_list = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Palette file '{palette_file}' not found.")

        self.palette = None
        if palette_number is not None:
            self.set_palette(palette_number)
            print(np.array(self.palette).shape)

    def set_palette(self, palette_number):
        """
        Sets the current palette from the loaded palette list.

        Args:
            palette_number (int): Index of the palette to set.
        Raises:
            IndexError: If the palette number is invalid.
        """
        try:
            self.palette = [hex_to_rgb(color) for color in self.palette_list[palette_number]]
        except IndexError:
            raise IndexError(f"Palette number {palette_number} is out of range.")

    def set_palette_from_image(self, image, num_colors=10):
        """
        Generates a palette from an image using KNN smoothing.

        Args:
            image: Input image (NumPy array or PIL Image).
            num_colors (int): Number of colors to extract. Defaults to 10.
        """
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
        arr = knn_smoothing(image, num_colors)
        self.palette = np.sort(np.unique(arr.reshape(-1, arr.shape[2]), axis=0), axis=0)


    def _create_image(self, size, interpolate=False):
        """
        Creates a visual representation of the current palette.

        Args:
            size (tuple): (height, width) of the output image.
            interpolate (bool): Whether to blend colors for gradients. Defaults to False.
        Returns:
            PIL.Image: Visualized palette.
        Raises:
            ValueError: If no palette is set.
        """
        if self.palette is None:
            raise ValueError("Palette not set. Call set_palette() first.")

        num_colors = len(self.palette)
        blocks = size[1] // (num_colors - 1 if interpolate else num_colors)
        color_image = np.zeros((size[0], size[1], 3), dtype=np.uint8)

        if interpolate:
            for i in range(num_colors - 1):
                r = np.linspace(self.palette[i][0], self.palette[i + 1][0], blocks, dtype=np.uint8)
                g = np.linspace(self.palette[i][1], self.palette[i + 1][1], blocks, dtype=np.uint8)
                b = np.linspace(self.palette[i][2], self.palette[i + 1][2], blocks, dtype=np.uint8)
                color_image[:, i * blocks:(i + 1) * blocks] = np.stack([r, g, b], axis=-1)

        else:
            for i in range(num_colors):
                color_image[:, i * blocks:(i + 1) * blocks] = self.palette[i]

        return Image.fromarray(color_image)

    def display_palette(self, size, interpolate=False):
        """
        Displays the current palette as an image.

        Args:
            size (tuple): (height, width) of the output image.
            interpolate (bool): Whether to interpolate colors. Defaults to False.
        Returns:
            PIL.Image: Palette visualization.
        """
        return self._create_image(size, interpolate)

    def get_palette_list_display(self, size, interpolate=False):
        """
        Generates visualizations for all palettes in the loaded list.

        Args:
            size (tuple): (height, width) of each palette image.
            interpolate (bool): Whether to interpolate colors. Defaults to False.
        Returns:
            list: List of PIL.Image objects representing all palettes.
        """
        re_l = []
        current_palette = self.palette
        for i in range(len(self.palette_list)):
            self.set_palette(i)
            re_l.append(self._create_image(size, interpolate))

        self.palette = current_palette
        return re_l