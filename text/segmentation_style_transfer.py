import numpy as np
from PIL import Image
import cv2

def segmentation_style_transfer(content_image, style_image, segmentation_mask, edge_smoothing=5):
    """
    Merges a content and style image based on the segmentation mask. Where the mask is True, will take pixels from style image.
    Where mask is False, will take pixels from content image.

    NOTE: expects content and style images to have same number of channels
    NOTE: when the dimensions of the content and style image differ, will assume these images had extra pixels added to the borders 
        due to CNN padding rounding, will crop out the middle box of the larger image for the merge

    Args:
        content_image (PIL Image): PIL image of original image
        style_image (PIL Image): PIL image of original image in new style
        segmentation_mask (NP array): True/False array of shape HxW
        edge_smoothing (Float): how much to smooth the intersections of the mask or not (default=5)

    Returns:
        merged_image (PIL Image): PIL image of the merged style and content images according to the mask
    """
    # convert images to np arrays
    content_array = np.array(content_image) # HxWx3
    style_array = np.array(style_image) # HxWx3

    # ensure dimensions of images are the same, crop the larger image smaller if not
    # can be different due to padding roundings during the creation of the style image
    c_H, c_W, c_C = content_array.shape
    s_H, s_W, s_C = style_array.shape
    if c_H < s_H: # if content image's height is smaller than style images
        left_offset = (s_H - c_H) // 2
        style_array = style_array[left_offset:left_offset+c_H, :, :] # reduce size of style image
    elif s_H < c_H: # if style image's height is smaller than content images
        left_offset = (c_H - s_H) // 2
        content_array = content_array[left_offset:left_offset+s_H, :, :] # reduce size of content image
        segmentation_mask = segmentation_mask[left_offset:left_offset+s_H, :] # and mask (since it was created based on content image)

    if c_W < s_W: # if content image's width is smaller than style images
        left_offset = (s_W - c_W) // 2
        style_array = style_array[:, left_offset:left_offset+c_W, :] # reduce size of style image
    elif s_W < c_W: # if style image's width is smaller than content images
        left_offset = (c_W - s_W) // 2
        content_array = content_array[:, left_offset:left_offset+s_W, :] # reduce size of content image
        segmentation_mask = segmentation_mask[:, left_offset:left_offset+s_W] # and mask (since it was created based on content image)
    
    # Merge the images
    if edge_smoothing:
        merged_array = _edge_smoothing(content_array, style_array, segmentation_mask, blur_strength=edge_smoothing)
    else:
        # broadcast segmentation mask to channels
        segmentation_mask_broadcasted = np.repeat(segmentation_mask[:, :, np.newaxis], c_C, axis=2)
        merged_array = np.where(segmentation_mask_broadcasted > 0, style_array, content_array)

    # Convert the merged array to a pil image
    merged_image = Image.fromarray(merged_array)

    return merged_image

def _edge_smoothing(content_array, style_array, segmentation_mask, blur_strength=5):
    """
    Merges a content and style array based on the segmentation mask. Where the mask is True, will take pixels from style image.
    Where mask is False, will take pixels from content image. Where edges of mask are, will smooth transition between pixels using gaussian blur.

    NOTE: expects content and style images to have same number of channels

    Args:
        content_array (NP array): array of original image (H x W x 3)
        style_array (NP array): array of original image in new style (H x W x 3)
        segmentation_mask (NP array): True/False array of shape HxW
        blur_strength (int): how strong the smoothing should be. NOTE: MUST BE ODD

    Returns:
        merged_array (NP array): array of the merged style and content images according to the mask
    """
    # Make sure blur strength is odd
    if blur_strength % 2 != 1:
        blur_strength += 1

    # Convert mask from True/False to 0 - 255 (for cv2)
    num_seg_mask = np.where(segmentation_mask, 1, 0).astype(np.uint8) * 255

    # Blur edges of mask
    blurred_mask = cv2.GaussianBlur(num_seg_mask, (blur_strength, blur_strength), 0)

    # Convert mask to 0 - 1 (for easy smooth multiplication)
    normalized_blurred_mask = blurred_mask / 255.0

    # Combine images according to smoothened mask
    # [..., None] allows broadcasting of mask to other channels in image
    # Convert to uint8 for PIL
    merged_array = (content_array * (1 - normalized_blurred_mask[..., None]) + style_array * normalized_blurred_mask[...,None]).astype(np.uint8)
    
    return merged_array