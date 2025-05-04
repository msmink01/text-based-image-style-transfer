import numpy as np
import cv2

def hex_to_rgb(hex_color):
    """Converts a hexadecimal color code to an RGB tuple.

    Args:
        hex_color (str): Color string in format '#RRGGBB' or 'RRGGBB'

    Returns:
        tuple: (R, G, B) values in 0-255 range
    """
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def knn_smoothing(img, k_clusters):
    """Reduces image colors using k-means clustering for visual simplification.

    Args:
        img (numpy.ndarray): Input image in BGR/RGB format
        k_clusters (int): Number of color clusters to create

    Returns:
        numpy.ndarray: Smoothed image with reduced color palette
    """
    pixel_values = img.reshape((-1, 3)).astype(np.float32)

    # Use less iterations for speed, might need adjustment
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixel_values, k_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    return centers[labels.flatten()].reshape(img.shape)

def edge_detection(img, edge_threshold):
    """Detects edges using Canny algorithm with dynamic threshold scaling.

    Args:
        img (numpy.ndarray): Input image
        edge_threshold (int): Base threshold value (0-100) scaled to 1-300 range

    Returns:
        numpy.ndarray: Binary edge mask with white edges on black background
    """
    edge_threshold = re_map(edge_threshold, 0, 100, 300, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, edge_threshold, edge_threshold * 2)
    return edges

def overlay_edges(img, edges):
    """Blends detected edges into image as black outlines.

    Args:
        img (numpy.ndarray): Original image
        edges (numpy.ndarray): Edge mask from edge_detection()

    Returns:
        numpy.ndarray: Combined image with edge overlay
    """
    edge_color = np.array([0, 0, 0])
    img = np.where(edges[..., None] > 0, edge_color, img)
    return img

def re_map(value, old_min, old_max, new_min, new_max):
    """Linear value remapping between numerical ranges.

    Args:
        value (float): Input value to remap
        old_min (float): Original range minimum
        old_max (float): Original range maximum
        new_min (float): Target range minimum
        new_max (float): Target range maximum

    Returns:
        float: Value mapped to new range
    """
    re = (value - old_min) * (new_max - new_min) / (old_max - old_min) + new_min

    if re > new_min:
        return new_min
    elif re < new_max:
        return new_max
    else:
        return re
