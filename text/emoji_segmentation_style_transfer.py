import numpy as np
from PIL import Image
import cv2
import math

def emoji_segmentation_style_transfer(content_image, style_image, segmentation_mask, emoji_mask, blur_strength=95, step_size_multiplier=0.5, style_strength=2):
    """
    Merges a content and style image based on the segmentation mask and emoji mask.
    Will first create an emoji style augmented content mask. Then will do use mask to merge images. 

    NOTE: expects content and style images to have same number of channels
    NOTE: when the dimensions of the content and style image differ, will assume these images had extra pixels added to the borders 
        due to CNN padding rounding, will crop out the middle box of the larger image for the merge

    Args:
        content_image (PIL Image): PIL image of original image
        style_image (PIL Image): PIL image of original image in new style
        segmentation_mask (NP array): True/False array of shape HxW
        emoji_mask (NP array): True/False array of shape 172x172 containing a style emoji
        blur_strength (int): how much outside the mask edges the emoji mask can go
        step_size_multiplier (float): the step size multiplier for the emoji mask
        style_strength (float): how strong the style emoji mask should be

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
    
    # Get the emoji augmented content mask
    merged_mask = _merge_content_style_segmentation_masks(segmentation_mask, emoji_mask, blur_strength=blur_strength, step_size_multiplier=step_size_multiplier)

    # Adjust merged mask according to the strength multiplier
    merged_mask = np.clip(merged_mask * style_strength, 0.0, 1.0)
    
    # Merge the images
    merged_array = (content_array * (1 - merged_mask[..., None]) + style_array * merged_mask[...,None]).astype(np.uint8)

    # Convert the merged array to a pil image
    merged_image = Image.fromarray(merged_array)

    return merged_image

def _merge_content_style_segmentation_masks(segmentation_mask, emoji_mask, blur_strength=95, step_size_multiplier=0.5):
    """
    Merges a content and style segmentation mask to create a style-driven content mask.

    NOTE: assumes segmentation_mask is larger than emoji_mask

    Args:
        segmentation_mask (NP array): True/False array of shape HxW
        emoji_mask (NP array): True/False array of shape 172x172 containing a style emoji
        blur_strength (int): how far emoji effect is allowed to go in mask

    Returns:
        merged_mask (NP array): array of the merged style and content images in range 0-1 of shape HxW
    """
    # Check blur strength is odd
    if blur_strength % 2 != 1:
        blur_strength += 1

    # Expand edges of seg mask to allow for some effects outside of original mask
    seg_mask = np.where(segmentation_mask, 1.0, 0.0)
    num_seg_mask = seg_mask.astype(np.uint8) * 255 # array of 255s/0s where True/False in seg mask
    blurred_mask = cv2.GaussianBlur(num_seg_mask, (blur_strength, blur_strength), 0)
    # Convert mask to 0 - 1 (for easy smooth multiplication)
    blurred_seg_mask = blurred_mask / 255.0 # array of [0, 1] around where seg mask was True
    H, W = blurred_seg_mask.shape

    # Convert emoji mask into 1s and 0s
    emo_mask = np.where(emoji_mask, 1.0, 0.0) # array of 1s/0s where True/False in emoji mask
    H_emo, W_emo = emo_mask.shape

    # Apply emoji_mask in different resolutions as a kernel across the segmentation mask; aggregate the results to get new segmentation mask
    merged_mask = np.zeros_like(seg_mask, dtype=float)
    res_scales = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for res_scale in res_scales:
        # Resize the emoji mask to the new scale size
        resized_emoji_mask = cv2.resize(emo_mask, (int(W_emo * res_scale), int(H_emo * res_scale)), interpolation=cv2.INTER_LINEAR)

        # Normalize resized emoji mask in the range [0, 1]
        resized_emoji_mask = resized_emoji_mask / np.max(resized_emoji_mask)

        # Get kernel width/height for this scale
        kernel_H, kernel_W = resized_emoji_mask.shape

        # Get the step size relative to the kernel width/height
        step_size_H = math.floor(kernel_H * step_size_multiplier)
        step_size_W = math.floor(kernel_W * step_size_multiplier)

        # Perform the sliding kernel operation with this new mask with the LEFT TOP POINT of emoji mask as anchor
        for i in range(0, H, step_size_H):
            for j in range(0, W, step_size_W):
                # Get current kernel region of original content mask, cut off bottom right if it goes out of bounds
                current_region = blurred_seg_mask[i:i+kernel_H, j:j+kernel_W]
                region_H, region_W = current_region.shape

                # Cut the emoji mask so it fits in this region (anchor top left corner, cut off bottom right)
                current_emoji_mask = resized_emoji_mask[:region_H, :region_W]

                # Apply current emoji kernel to current region
                to_add = current_region * current_emoji_mask
                merged_mask[i:i + kernel_H, j:j + kernel_W] += to_add / (np.max(to_add) + 1e-7)

        # Perform the sliding kernel operation with this new mask with the RIGHT TOP POINT of emoji mask as anchor
        for i in range(0, H, step_size_H):
            for j in range(W, 1, -step_size_W):
                # Get current kernel region of original content mask, cut off bottom left if it goes out of bounds
                region_left = j-kernel_H if j-kernel_H > 0 else 0
                current_region = blurred_seg_mask[i:i+kernel_H, region_left:j]
                region_H, region_W = current_region.shape

                # Cut the emoji mask so it fits in this region (anchor top right corner, cut off bottom left)
                emoji_left = kernel_W-region_W if kernel_W-region_W > 0 else 0
                current_emoji_mask = resized_emoji_mask[:region_H, emoji_left:]

                # Apply current emoji kernel to current region
                to_add = current_region * current_emoji_mask
                merged_mask[i:i + kernel_H, region_left:j] += to_add / (np.max(to_add) + 1e-7)

        # Perform the sliding kernel operation with this new mask with the LEFT BOTTOM POINT of emoji mask as anchor
        for i in range(H, 1, -step_size_H):
            for j in range(0, W, step_size_W):
                # Get current kernel region of original content mask, cut off top right if it goes out of bounds
                region_top = i-kernel_H if i-kernel_H > 0 else 0
                current_region = blurred_seg_mask[region_top:i, j:j+kernel_W]
                region_H, region_W = current_region.shape

                # Cut the emoji mask so it fits in this region (anchor bottom left corner, cut off top right)
                emoji_top = kernel_H-region_H if kernel_H-region_H > 0 else 0
                current_emoji_mask = resized_emoji_mask[emoji_top:, :region_W]

                # Apply current emoji kernel to current region
                to_add = current_region * current_emoji_mask
                merged_mask[region_top:i, j:j + kernel_W] += to_add / (np.max(to_add) + 1e-7)

        # Perform the sliding kernel operation with this new mask with the RIGHT BOTTOM POINT of emoji mask as anchor
        for i in range(H, 1, -step_size_H):
            for j in range(W, 1, -step_size_W):
                # Get current kernel region of original content mask, cut off top left if it goes out of bounds
                region_top = i-kernel_H if i-kernel_H > 0 else 0
                region_left = j-kernel_H if j-kernel_H > 0 else 0
                current_region = blurred_seg_mask[region_top:i, region_left:j]
                region_H, region_W = current_region.shape

                # Cut the emoji mask so it fits in this region (anchor bottom right corner, cut off top left)
                emoji_top = kernel_H-region_H if kernel_H-region_H > 0 else 0
                emoji_left = kernel_W-region_W if kernel_W-region_W > 0 else 0
                current_emoji_mask = resized_emoji_mask[emoji_top:, emoji_left:]

                # Apply current emoji kernel to current region
                to_add = current_region * current_emoji_mask
                merged_mask[region_top:i, region_left:j] += to_add / (np.max(to_add) + 1e-7)

    # Normalize the final mask so in the range [0, 1]
    merged_mask = merged_mask / np.max(merged_mask)
    
    return merged_mask