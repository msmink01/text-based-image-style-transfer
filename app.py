"""
Interactive Media Processing Application
----------------------------------------
A Gradio-based UI for applying advanced image/video processing effects including:
- Neural Style Transfer
- Pixel Art Conversion
- Depth-aware Stylization
- Text-guided Effects
- Color Palette Transfer
- Video Processing with Interpolation

Key Components:
1. Model Initialization - Loads all AI models upfront
2. Processing Functions - Core image/video transformation logic
3. UI Components - Interactive Gradio interface
4. Visibility Handlers - Dynamic UI element management
5. Utility Functions - Supporting operations and helpers
"""

import math
import os
import tempfile
import time

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

from color_palette.ColorPaletteTransfer import ColorPaletteTransfer
from components.pixel_art.colour_palette import ColourPalette
from components.pixel_art.pixel_art import PixelArt
from components.style_transfer_depth.style_transfer_depth import DepthStyle
from multi_style_transfer.run_style_transfer import run_multi_style_transfer
from text.EmojiMaskExtractor import EmojiMaskExtractor
# Set up text related imports and models
from text.FastTextTransfer import FastTextStyleTransfer
from text.TextMaskExtractor import TextMaskExtractor
from text.emoji_segmentation_style_transfer import _merge_content_style_segmentation_masks, \
    emoji_segmentation_style_transfer
from text.segmentation_style_transfer import segmentation_style_transfer


# ======================
# 1. MODEL INITIALIZATION
# ======================
# Load all AI models and processing pipelines during startup
now = time.time()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")
text_transfer_model = FastTextStyleTransfer(device)
print("Fast Text Style Transfer model loaded successfully!")

mask_extractor = TextMaskExtractor(device)
print("Mask extractor loaded successfully!")

emoji_mask_extractor = EmojiMaskExtractor(device)
print("Emoji mask extractor loaded successfully!")

# Set up temporary directory used for processed videos
video_temp_dir = tempfile.TemporaryDirectory()
print(f"Temporary directory for processed videos: {video_temp_dir.name}")

pixel_art = PixelArt()
colour_palette = ColourPalette()
size = [30, 300]
colour_palette_list = colour_palette.get_palette_list_display(size)
colour_palette_list_interpolate = colour_palette.get_palette_list_display(size, interpolate=True)
print("Pixel art loaded successfully!")

depth_pipline = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
depth_style = DepthStyle(device)
print("Depth models loaded successfully!")

print(f"Time taken to load all models: {time.time() - now:.2f} seconds")


# ======================
# 2. CORE PROCESSING
# ======================

# List of available MAIN effects
list_of_effects = ["Convert Output to Grayscale", "Text-Based Effects", "Pixel Art", "Style Transfer", "Style Mixing", "Color Palette Transfer", "Depth Based Style Transfer"]
text = list_of_effects[1]
pixel = list_of_effects[2]
style = [list_of_effects[3], list_of_effects[6]]
mixed_style = list_of_effects[4]
color_palette = list_of_effects[5]
depth = list_of_effects[6]

# List of available text effects (visible if text style is selected)
list_of_text_effects = [
    "Text-Based Style Transfer",  # perform style transfer: no masking
    "Location Masking",  # content mask only
    "Style Masking",  # emoji style mask only
]
text_style_transfer = list_of_text_effects[0]
text_location_masking = list_of_text_effects[1]
text_style_masking = list_of_text_effects[2]

list_pixel_art_effects = ["Colour Palette", "Edges"]
pixel_colour = list_pixel_art_effects[0]
pixel_edge = list_pixel_art_effects[1]


list_of_depth_effects = ["Modified loss Style Transfer", "Multi Plane Image Style Transfer"]
depth_style_transfer = list_of_depth_effects[0]
depth_multi_plane = list_of_depth_effects[1]

# Used to enable/disable channel attention for style mixing
CHANNEL_ATT_ENABLED = False

def apply_image_process(image_filepath, checkbox_values, input_style=None,
                        text_checkbox_values=None, text_box=None, text_location_box=None, text_style_masking_box=None, text_masked_transfer_edge_smoothing=None, text_emoji_blur_strength=None, text_emoji_step_size=None, text_masked_style_strength=None,
                        p_size_slider=0.4, p_checkbox=list,p_colour_dropbox=0, p_colour_interpolate=False, p_edge_slider=50, p_select_im=False, p_in=None, p_in_slid=10,
                        style_img_weight=None, style_image1=None, style_image2=None, color_palette_style=None,
                        d_check_box=None, depth_mip_n=2):


    """
    Apply the selected image processing effects.

    Args:
        image_filepath (str): Input image filepath.
        checkbox_values (list): Selected main effects.
        input_style (PIL.Image, optional): Style image.
        text_checkbox_values (list, optional): Selected text effects
        text_box (str, optional): Selected text style
        text_location_box (str, optional): Selected text mask
        text_style_masking_box (str, optional): Selected text style mask
        text_masked_transfer_edge_smoothing (int, optional): how much to smooth the edges of the content mask
        text_emoji_blur_strength (int, optional):  how much to allow the style mask to escape the content mask
        text_emoji_step_size (float, optional): how often to apply the style mask
        text_masked_style_strength (float, optional): how strong the style should be applied
        p_size_slider (float, optional): Slider value for pixel size
        p_checkbox (list, optional): Selected pixel art effects
        p_colour_dropbox (int, optional): Selected colour palette
        p_colour_interpolate (bool, optional): Interpolate colours
        p_edge_slider (int, optional): Edge threshold
        p_select_im (bool, optional): Select image to use as colour palette
        p_in (PIL.Image, optional): Image to use as colour palette
        p_in_slid (int, optional): Number of colour palette to get from image

    Returns:
        PIL.Image: Processed output image in "L" or "RGB" format
    """
    # Load image in from filepath
    image = None
    output_image = None
    if image_filepath:
        image = Image.open(image_filepath)
        output_image = image.copy()

    if list_of_effects[0] in checkbox_values:
        # Convert output to grayscale
        output_image = output_image.convert("L")

    if list_of_effects[1] in checkbox_values:  # if text-based is selected
        if not image and list_of_text_effects[1] not in text_checkbox_values and list_of_text_effects[2] not in text_checkbox_values:
            # No input image, no input content mask, no input style mask: return nothing
            return None

        if image and list_of_text_effects[0] in text_checkbox_values and list_of_text_effects[
            1] in text_checkbox_values and list_of_text_effects[2] in text_checkbox_values:
            # We have an input image, want to do style transfer, have a content mask prompt, and a style mask prompt
            # Do content and style masked style transfer!
            style_transfer_prompt = text_box
            content_mask_prompt = text_location_box
            style_mask_prompt = text_style_masking_box
            blur_strength = text_emoji_blur_strength if text_emoji_blur_strength else 0
            step_size_multiplier = text_emoji_step_size if text_emoji_step_size else 0
            style_strength = text_masked_style_strength if text_masked_style_strength else 0

            if not style_transfer_prompt or not content_mask_prompt or not style_mask_prompt:
                # Ensure prompts are complete
                return None

            mask = mask_extractor.perform_mask_extraction(image_filepath, content_mask_prompt)
            emoji_mask = emoji_mask_extractor.perform_emoji_mask_extraction(style_mask_prompt)
            processed_img = text_transfer_model.perform_transfer(image, style_transfer_prompt)
            output_image = emoji_segmentation_style_transfer(image, processed_img, mask, emoji_mask,
                                                             blur_strength=blur_strength,
                                                             step_size_multiplier=step_size_multiplier,
                                                             style_strength=style_strength)

        elif image and list_of_text_effects[0] in text_checkbox_values and list_of_text_effects[
            1] in text_checkbox_values and list_of_text_effects[2] not in text_checkbox_values:
            # We have an input image, want to do style transfer, have a content mask prompt, and don't have a style mask prompt
            # Do content masked style transfer!
            style_transfer_prompt = text_box
            content_mask_prompt = text_location_box
            edge_smoothing = text_masked_transfer_edge_smoothing if text_masked_transfer_edge_smoothing else 0

            if not style_transfer_prompt or not content_mask_prompt:
                # Ensure prompts are complete
                return None

            mask = mask_extractor.perform_mask_extraction(image_filepath, content_mask_prompt)
            processed_img = text_transfer_model.perform_transfer(image, style_transfer_prompt)
            output_image = segmentation_style_transfer(image, processed_img, mask, edge_smoothing=edge_smoothing)

        elif image and list_of_text_effects[0] in text_checkbox_values and list_of_text_effects[
            1] not in text_checkbox_values and list_of_text_effects[2] in text_checkbox_values:
            # We have an input image, want to do style transfer, don't have a content mask prompt, and have a style mask prompt
            # Do content masked style transfer!
            style_transfer_prompt = text_box
            style_mask_prompt = text_style_masking_box
            blur_strength = text_emoji_blur_strength if text_emoji_blur_strength else 0
            step_size_multiplier = text_emoji_step_size if text_emoji_step_size else 0
            style_strength = text_masked_style_strength if text_masked_style_strength else 0

            if not style_transfer_prompt or not style_mask_prompt:
                # Ensure prompts are complete
                return None

            emoji_mask = emoji_mask_extractor.perform_emoji_mask_extraction(style_mask_prompt)
            processed_img = text_transfer_model.perform_transfer(image, style_transfer_prompt)
            mask = np.ones_like(processed_img)[:, :, 0]
            output_image = emoji_segmentation_style_transfer(image, processed_img, mask, emoji_mask,
                                                             blur_strength=blur_strength,
                                                             step_size_multiplier=step_size_multiplier,
                                                             style_strength=style_strength)

        elif image and list_of_text_effects[0] in text_checkbox_values and list_of_text_effects[
            1] not in text_checkbox_values and list_of_text_effects[2] not in text_checkbox_values:
            # We have an input image, want to do style transfer, don't have a content mask prompt, and don't have a style mask prompt
            # Do unmasked style transfer!
            style_transfer_prompt = text_box

            if not style_transfer_prompt:
                # Ensure prompts are complete
                return None

            output_image = text_transfer_model.perform_transfer(image, style_transfer_prompt)

        elif image and list_of_text_effects[0] not in text_checkbox_values and list_of_text_effects[
            1] in text_checkbox_values and list_of_text_effects[2] not in text_checkbox_values:
            # We have an input image, don't want to do style transfer, we do have a content mask prompt, and don't have a style mask prompt
            # Get the content mask by itself!
            content_mask_prompt = text_location_box

            if not content_mask_prompt:
                # Ensure prompts are complete
                return None

            mask = mask_extractor.perform_mask_extraction(image_filepath, content_mask_prompt)
            output_image = Image.fromarray(mask).convert("L")

        elif list_of_text_effects[0] not in text_checkbox_values and list_of_text_effects[
            1] not in text_checkbox_values and list_of_text_effects[2] in text_checkbox_values:
            # We may or may not have an input image, don't want to do style transfer, we don't have a content mask prompt, and do have a style mask prompt
            # Get the emoji mask by itself!
            style_mask_prompt = text_style_masking_box

            if not style_mask_prompt:
                # Ensure prompts are complete
                return None

            mask = emoji_mask_extractor.perform_emoji_mask_extraction(style_mask_prompt)
            output_image = Image.fromarray(mask.astype(np.uint8) * 255).convert("L")

        elif image and list_of_text_effects[0] not in text_checkbox_values and list_of_text_effects[
            1] in text_checkbox_values and list_of_text_effects[2] in text_checkbox_values:
            # We have an input image, don't want to do style transfer, we do have a content mask prompt, and do have a style mask prompt
            # Create the emoji/location merged mask!
            content_mask_prompt = text_location_box
            style_mask_prompt = text_style_masking_box
            blur_strength = text_emoji_blur_strength if text_emoji_blur_strength else 0
            step_size_multiplier = text_emoji_step_size if text_emoji_step_size else 0

            if not content_mask_prompt or not style_mask_prompt:
                # Ensure prompts are complete
                return None

            mask = mask_extractor.perform_mask_extraction(image_filepath, content_mask_prompt)
            emoji_mask = emoji_mask_extractor.perform_emoji_mask_extraction(style_mask_prompt)
            merged_mask = _merge_content_style_segmentation_masks(mask, emoji_mask, blur_strength=blur_strength,
                                                                  step_size_multiplier=step_size_multiplier)
            output_image = Image.fromarray((merged_mask * 255).astype(np.uint8)).convert("L")

    if list_of_effects[2] in checkbox_values and image:

        if list_pixel_art_effects[0] in p_checkbox:
            if p_select_im:
                colour_palette_input = ColourPalette()
                colour_palette_input.set_palette_from_image(p_in, p_in_slid)
            else:
                colour_palette_input = ColourPalette(palette_number=p_colour_dropbox)
        else:
            colour_palette_input = None

        if list_pixel_art_effects[1] in p_checkbox:
            edge_detect = True
            if p_edge_slider == 0:
                edge_detect = False
        else:
            edge_detect = False

        if list_of_effects[1] in checkbox_values: # user has selected text-based as well as pixel art
            if list_of_text_effects[1] in text_checkbox_values and list_of_text_effects[0] not in text_checkbox_values and list_of_text_effects[2] not in text_checkbox_values:
                # Want to do text based masking without texture or transfer
                content_mask_prompt = text_location_box
                edge_smoothing = text_masked_transfer_edge_smoothing if text_masked_transfer_edge_smoothing else 5
                if not content_mask_prompt:
                    # Ensure prompts are complete
                    return None
                
                # Get mask
                mask = mask_extractor.perform_mask_extraction(image_filepath, content_mask_prompt)
                # Get overall pixel art image from ORIGINAL image
                pixel_image = pixel_art.process(image, pixel_size=p_size_slider, colour_palette=colour_palette_input,
                                                interpolate=p_colour_interpolate, edge_detect=edge_detect,
                                                edge_threshold=p_edge_slider)
                # Merge original image with pixel art image
                output_image = segmentation_style_transfer(image, pixel_image, mask, edge_smoothing=edge_smoothing)
            elif list_of_text_effects[2] in text_checkbox_values and list_of_text_effects[0] not in text_checkbox_values and list_of_text_effects[1] not in text_checkbox_values:
                # Want to do text based texture masking without location
                style_mask_prompt = text_style_masking_box
                blur_strength = text_emoji_blur_strength if text_emoji_blur_strength else 95
                step_size_multiplier = text_emoji_step_size if text_emoji_step_size else 0.5
                style_strength = text_masked_style_strength if text_masked_style_strength else 1.5
                if not style_mask_prompt:
                    # Ensure prompts are complete
                    return None

                # Get overall pixel art image from ORIGINAL image
                pixel_image = pixel_art.process(image, pixel_size=p_size_slider, colour_palette=colour_palette_input,
                                                interpolate=p_colour_interpolate, edge_detect=edge_detect,
                                                edge_threshold=p_edge_slider)
                # Get texture mask
                emoji_mask = emoji_mask_extractor.perform_emoji_mask_extraction(style_mask_prompt)
                mask = np.ones_like(pixel_image)[:, :, 0]
                output_image = emoji_segmentation_style_transfer(image, pixel_image, mask, emoji_mask,
                                                                blur_strength=blur_strength,
                                                                step_size_multiplier=step_size_multiplier,
                                                                style_strength=style_strength)
            elif list_of_text_effects[2] in text_checkbox_values and list_of_text_effects[1] in text_checkbox_values and list_of_text_effects[0] not in text_checkbox_values:
                # Want to do text based location masking with texture!
                content_mask_prompt = text_location_box
                style_mask_prompt = text_style_masking_box
                blur_strength = text_emoji_blur_strength if text_emoji_blur_strength else 95
                step_size_multiplier = text_emoji_step_size if text_emoji_step_size else 0.5
                style_strength = text_masked_style_strength if text_masked_style_strength else 1.5
                if not content_mask_prompt or not style_mask_prompt:
                    # Ensure prompts are complete
                    return None
                
                # Get overall pixel art image from ORIGINAL image
                pixel_image = pixel_art.process(image, pixel_size=p_size_slider, colour_palette=colour_palette_input,
                                                interpolate=p_colour_interpolate, edge_detect=edge_detect,
                                                edge_threshold=p_edge_slider)

                mask = mask_extractor.perform_mask_extraction(image_filepath, content_mask_prompt)
                emoji_mask = emoji_mask_extractor.perform_emoji_mask_extraction(style_mask_prompt)
                output_image = emoji_segmentation_style_transfer(image, pixel_image, mask, emoji_mask,
                                                                blur_strength=blur_strength,
                                                                step_size_multiplier=step_size_multiplier,
                                                                style_strength=style_strength)
            else:
                # Text transfer was done on the image, simply do pixel art on the completed text image!
                output_image = pixel_art.process(output_image, pixel_size=p_size_slider, colour_palette=colour_palette_input,
                                                interpolate=p_colour_interpolate, edge_detect=edge_detect,
                                                edge_threshold=p_edge_slider)
        else: # text-based was not selected, do normal pixel art on whatever image appears
            output_image = pixel_art.process(output_image, pixel_size=p_size_slider, colour_palette=colour_palette_input,
                                                interpolate=p_colour_interpolate, edge_detect=edge_detect,
                                                edge_threshold=p_edge_slider)

    # Apply style image effect
    if list_of_effects[3] in checkbox_values and input_style:
        # Implement style transfer
        # Define the channel-wise mean and standard deviation used for VGG training
        vgg_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        vgg_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

        # Define hyperparameters with which to run the (multi-)style transfer
        num_steps = 400
        random_init = False
        w_style = 5e5
        w_content = 1
        w_tv = 2e1
        w_edge = 2e1
        channel_attention = False  # whether to apply channel attention when computing the content loss

        if output_image and input_style:
            if list_of_effects[1] in checkbox_values: # user has selected text-based as well as style transfer
                if list_of_text_effects[1] in text_checkbox_values and list_of_text_effects[0] not in text_checkbox_values and list_of_text_effects[2] not in text_checkbox_values:
                    # Want to do text based masking without texture or transfer
                    content_mask_prompt = text_location_box
                    edge_smoothing = text_masked_transfer_edge_smoothing if text_masked_transfer_edge_smoothing else 5
                    if not content_mask_prompt:
                        # Ensure prompts are complete
                        return None
                    
                    # Get mask
                    mask = mask_extractor.perform_mask_extraction(image_filepath, content_mask_prompt)
                    # Get overall transfer image from ORIGINAL image
                    style_image = run_multi_style_transfer(vgg_mean, vgg_std, image,
                                                    num_steps=num_steps, random_init=random_init, w_style=w_style,
                                                    w_content=w_content, w_tv=w_tv, w_edge=w_edge,
                                                    channel_attention=channel_attention, style_img1=input_style,
                                                    device=device)
                    # Merge original image with pixel art image
                    output_image = segmentation_style_transfer(image, style_image, mask, edge_smoothing=edge_smoothing)
                elif list_of_text_effects[2] in text_checkbox_values and list_of_text_effects[0] not in text_checkbox_values and list_of_text_effects[1] not in text_checkbox_values:
                    # Want to do text based texture masking without location
                    style_mask_prompt = text_style_masking_box
                    blur_strength = text_emoji_blur_strength if text_emoji_blur_strength else 95
                    step_size_multiplier = text_emoji_step_size if text_emoji_step_size else 0.5
                    style_strength = text_masked_style_strength if text_masked_style_strength else 1.5
                    if not style_mask_prompt:
                        # Ensure prompts are complete
                        return None

                    # Get overall transfer image from ORIGINAL image
                    style_image = run_multi_style_transfer(vgg_mean, vgg_std, image,
                                                    num_steps=num_steps, random_init=random_init, w_style=w_style,
                                                    w_content=w_content, w_tv=w_tv, w_edge=w_edge,
                                                    channel_attention=channel_attention, style_img1=input_style,
                                                    device=device)
                    # Get texture mask
                    emoji_mask = emoji_mask_extractor.perform_emoji_mask_extraction(style_mask_prompt)
                    mask = np.ones_like(style_image)[:, :, 0]
                    output_image = emoji_segmentation_style_transfer(image, style_image, mask, emoji_mask,
                                                                    blur_strength=blur_strength,
                                                                    step_size_multiplier=step_size_multiplier,
                                                                    style_strength=style_strength)
                elif list_of_text_effects[2] in text_checkbox_values and list_of_text_effects[1] in text_checkbox_values and list_of_text_effects[0] not in text_checkbox_values:
                    # Want to do text based location masking with texture!
                    content_mask_prompt = text_location_box
                    style_mask_prompt = text_style_masking_box
                    blur_strength = text_emoji_blur_strength if text_emoji_blur_strength else 95
                    step_size_multiplier = text_emoji_step_size if text_emoji_step_size else 0.5
                    style_strength = text_masked_style_strength if text_masked_style_strength else 1.5
                    if not content_mask_prompt or not style_mask_prompt:
                        # Ensure prompts are complete
                        return None
                    
                    # Get overall transfer image from ORIGINAL image
                    style_image = run_multi_style_transfer(vgg_mean, vgg_std, image,
                                                    num_steps=num_steps, random_init=random_init, w_style=w_style,
                                                    w_content=w_content, w_tv=w_tv, w_edge=w_edge,
                                                    channel_attention=channel_attention, style_img1=input_style,
                                                    device=device)

                    mask = mask_extractor.perform_mask_extraction(image_filepath, content_mask_prompt)
                    emoji_mask = emoji_mask_extractor.perform_emoji_mask_extraction(style_mask_prompt)
                    output_image = emoji_segmentation_style_transfer(image, style_image, mask, emoji_mask,
                                                                    blur_strength=blur_strength,
                                                                    step_size_multiplier=step_size_multiplier,
                                                                    style_strength=style_strength)
                else:
                    # Text transfer was done on the image, simply do style transfer on the completed text image!
                    output_image = run_multi_style_transfer(vgg_mean, vgg_std, output_image,
                                                    num_steps=num_steps, random_init=random_init, w_style=w_style,
                                                    w_content=w_content, w_tv=w_tv, w_edge=w_edge,
                                                    channel_attention=channel_attention, style_img1=input_style,
                                                    device=device)
            else: # text-based was not selected, do normal style transfer on whatever image appears
                output_image = run_multi_style_transfer(vgg_mean, vgg_std, output_image,
                                                    num_steps=num_steps, random_init=random_init, w_style=w_style,
                                                    w_content=w_content, w_tv=w_tv, w_edge=w_edge,
                                                    channel_attention=channel_attention, style_img1=input_style,
                                                    device=device)
        else:
            # If either the content or the style image is missing, do not apply style transfer
            return None

    if list_of_effects[4] in checkbox_values:
        # Implement style mixing
        # Define the channel-wise mean and standard deviation used for VGG training
        vgg_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        vgg_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

        # Define hyperparameters with which to run the (multi-)style transfer
        num_steps = 400
        random_init = False
        w_style = 5e5
        w_content = 1
        w_tv = 2e1
        w_edge = 2e1
        channel_attention = CHANNEL_ATT_ENABLED # whether to apply channel attention when computing the content loss

        if output_image and style_image1 and style_image2 and style_img_weight:
            # With 2 style images we can have multi-style transfer
            style1 = Image.open(style_image1)
            style2 = Image.open(style_image2)

            style_weight = style_img_weight

            if list_of_effects[1] in checkbox_values: # user has selected text-based as well as style mixing
                if list_of_text_effects[1] in text_checkbox_values and list_of_text_effects[0] not in text_checkbox_values and list_of_text_effects[2] not in text_checkbox_values:
                    # Want to do text based masking without texture or transfer
                    content_mask_prompt = text_location_box
                    edge_smoothing = text_masked_transfer_edge_smoothing if text_masked_transfer_edge_smoothing else 5
                    if not content_mask_prompt:
                        # Ensure prompts are complete
                        return None
                    
                    # Get mask
                    mask = mask_extractor.perform_mask_extraction(image_filepath, content_mask_prompt)
                    # Get overall transfer image from ORIGINAL image
                    style_image = run_multi_style_transfer(vgg_mean, vgg_std, image,
                                                    num_steps=num_steps, random_init=random_init, w_style=w_style,
                                                    w_content=w_content, w_tv=w_tv, w_edge=w_edge,
                                                    channel_attention=channel_attention, style_img1=style1, style_img2=style2,
                                                    style_img_weight=style_weight, device=device)
                    # Merge original image with pixel art image
                    output_image = segmentation_style_transfer(image, style_image, mask, edge_smoothing=edge_smoothing)
                elif list_of_text_effects[2] in text_checkbox_values and list_of_text_effects[0] not in text_checkbox_values and list_of_text_effects[1] not in text_checkbox_values:
                    # Want to do text based texture masking without location
                    style_mask_prompt = text_style_masking_box
                    blur_strength = text_emoji_blur_strength if text_emoji_blur_strength else 95
                    step_size_multiplier = text_emoji_step_size if text_emoji_step_size else 0.5
                    style_strength = text_masked_style_strength if text_masked_style_strength else 1.5
                    if not style_mask_prompt:
                        # Ensure prompts are complete
                        return None

                    # Get overall transfer image from ORIGINAL image
                    style_image = run_multi_style_transfer(vgg_mean, vgg_std, image,
                                                    num_steps=num_steps, random_init=random_init, w_style=w_style,
                                                    w_content=w_content, w_tv=w_tv, w_edge=w_edge,
                                                    channel_attention=channel_attention, style_img1=style1, style_img2=style2,
                                                    style_img_weight=style_weight, device=device)
                    # Get texture mask
                    emoji_mask = emoji_mask_extractor.perform_emoji_mask_extraction(style_mask_prompt)
                    mask = np.ones_like(style_image)[:, :, 0]
                    output_image = emoji_segmentation_style_transfer(image, style_image, mask, emoji_mask,
                                                                    blur_strength=blur_strength,
                                                                    step_size_multiplier=step_size_multiplier,
                                                                    style_strength=style_strength)
                elif list_of_text_effects[2] in text_checkbox_values and list_of_text_effects[1] in text_checkbox_values and list_of_text_effects[0] not in text_checkbox_values:
                    # Want to do text based location masking with texture!
                    content_mask_prompt = text_location_box
                    style_mask_prompt = text_style_masking_box
                    blur_strength = text_emoji_blur_strength if text_emoji_blur_strength else 95
                    step_size_multiplier = text_emoji_step_size if text_emoji_step_size else 0.5
                    style_strength = text_masked_style_strength if text_masked_style_strength else 1.5
                    if not content_mask_prompt or not style_mask_prompt:
                        # Ensure prompts are complete
                        return None
                    
                    # Get overall transfer image from ORIGINAL image
                    style_image = run_multi_style_transfer(vgg_mean, vgg_std, image,
                                                    num_steps=num_steps, random_init=random_init, w_style=w_style,
                                                    w_content=w_content, w_tv=w_tv, w_edge=w_edge,
                                                    channel_attention=channel_attention, style_img1=style1, style_img2=style2,
                                                    style_img_weight=style_weight, device=device)

                    mask = mask_extractor.perform_mask_extraction(image_filepath, content_mask_prompt)
                    emoji_mask = emoji_mask_extractor.perform_emoji_mask_extraction(style_mask_prompt)
                    output_image = emoji_segmentation_style_transfer(image, style_image, mask, emoji_mask,
                                                                    blur_strength=blur_strength,
                                                                    step_size_multiplier=step_size_multiplier,
                                                                    style_strength=style_strength)
                else:
                    # Text transfer was done on the image, simply do style mixing on the completed text image!
                    output_image = run_multi_style_transfer(vgg_mean, vgg_std, output_image,
                                                    num_steps=num_steps, random_init=random_init, w_style=w_style,
                                                    w_content=w_content, w_tv=w_tv, w_edge=w_edge,
                                                    channel_attention=channel_attention, style_img1=style1, style_img2=style2,
                                                    style_img_weight=style_weight, device=device)
            else: # text-based was not selected, do normal style transfer on whatever image appears
                output_image = run_multi_style_transfer(vgg_mean, vgg_std, output_image,
                                                    num_steps=num_steps, random_init=random_init, w_style=w_style,
                                                    w_content=w_content, w_tv=w_tv, w_edge=w_edge,
                                                    channel_attention=channel_attention, style_img1=style1, style_img2=style2,
                                                    style_img_weight=style_weight, device=device)
            
        else:
            # Default to single-style transfer with the one input style
            if style_image1 and style_img_weight:
                style = Image.open(style_image1)
                style_weight = style_img_weight
            elif style_image2 and style_img_weight:
                style = Image.open(style_image2)
                style_weight = style_img_weight
            else:
                # If no style image is chosen, or if the weight is missing, return None
                return None

            output_image = run_multi_style_transfer(vgg_mean, vgg_std, output_image,
                                                    num_steps=num_steps, random_init=random_init, w_style=w_style,
                                                    w_content=w_content, w_tv=w_tv, w_edge=w_edge,
                                                    channel_attention=channel_attention, style_img1=style,
                                                    style_img_weight=style_weight, device=device)

    if list_of_effects[5] in checkbox_values:
        # Implement color palette transfer
        color_palette_transfer = ColorPaletteTransfer()
        if output_image and color_palette_style:
            if list_of_effects[1] in checkbox_values: # user has selected text-based as well as color palette
                if list_of_text_effects[1] in text_checkbox_values and list_of_text_effects[0] not in text_checkbox_values and list_of_text_effects[2] not in text_checkbox_values:
                    # Want to do text based masking without texture or transfer
                    content_mask_prompt = text_location_box
                    edge_smoothing = text_masked_transfer_edge_smoothing if text_masked_transfer_edge_smoothing else 5
                    if not content_mask_prompt:
                        # Ensure prompts are complete
                        return None
                    
                    # Get mask
                    mask = mask_extractor.perform_mask_extraction(image_filepath, content_mask_prompt)
                    # Get overall transfer image from ORIGINAL image
                    palette_image = color_palette_transfer.color_transfer_pipeline(image, color_palette_style)
                    # Merge original image with pixel art image
                    output_image = segmentation_style_transfer(image, palette_image, mask, edge_smoothing=edge_smoothing)
                elif list_of_text_effects[2] in text_checkbox_values and list_of_text_effects[0] not in text_checkbox_values and list_of_text_effects[1] not in text_checkbox_values:
                    # Want to do text based texture masking without location
                    style_mask_prompt = text_style_masking_box
                    blur_strength = text_emoji_blur_strength if text_emoji_blur_strength else 95
                    step_size_multiplier = text_emoji_step_size if text_emoji_step_size else 0.5
                    style_strength = text_masked_style_strength if text_masked_style_strength else 1.5
                    if not style_mask_prompt:
                        # Ensure prompts are complete
                        return None

                    # Get overall transfer image from ORIGINAL image
                    palette_image = color_palette_transfer.color_transfer_pipeline(image, color_palette_style)

                    # Get texture mask
                    emoji_mask = emoji_mask_extractor.perform_emoji_mask_extraction(style_mask_prompt)
                    mask = np.ones_like(style_image)[:, :, 0]
                    output_image = emoji_segmentation_style_transfer(image, palette_image, mask, emoji_mask,
                                                                    blur_strength=blur_strength,
                                                                    step_size_multiplier=step_size_multiplier,
                                                                    style_strength=style_strength)
                elif list_of_text_effects[2] in text_checkbox_values and list_of_text_effects[1] in text_checkbox_values and list_of_text_effects[0] not in text_checkbox_values:
                    # Want to do text based location masking with texture!
                    content_mask_prompt = text_location_box
                    style_mask_prompt = text_style_masking_box
                    blur_strength = text_emoji_blur_strength if text_emoji_blur_strength else 95
                    step_size_multiplier = text_emoji_step_size if text_emoji_step_size else 0.5
                    style_strength = text_masked_style_strength if text_masked_style_strength else 1.5
                    if not content_mask_prompt or not style_mask_prompt:
                        # Ensure prompts are complete
                        return None
                    
                    # Get overall transfer image from ORIGINAL image
                    palette_image = color_palette_transfer.color_transfer_pipeline(image, color_palette_style)

                    mask = mask_extractor.perform_mask_extraction(image_filepath, content_mask_prompt)
                    emoji_mask = emoji_mask_extractor.perform_emoji_mask_extraction(style_mask_prompt)
                    output_image = emoji_segmentation_style_transfer(image, palette_image, mask, emoji_mask,
                                                                    blur_strength=blur_strength,
                                                                    step_size_multiplier=step_size_multiplier,
                                                                    style_strength=style_strength)
                else:
                    # Text transfer was done on the image, simply do palette transfer on the completed text image!
                    output_image = color_palette_transfer.color_transfer_pipeline(output_image, color_palette_style)
            else: # text-based was not selected, do normal palette transfer on whatever image appears
                output_image = color_palette_transfer.color_transfer_pipeline(output_image, color_palette_style)
        else:
            # If either the input image or the style image is missing, return None
            return None

    if list_of_effects[6] in checkbox_values and input_style:
        if list_of_effects[1] in checkbox_values: # user has selected text-based as well as depth style transfer
            if list_of_text_effects[1] in text_checkbox_values and list_of_text_effects[0] not in text_checkbox_values and list_of_text_effects[2] not in text_checkbox_values:
                # Want to do text based masking without texture or transfer
                content_mask_prompt = text_location_box
                edge_smoothing = text_masked_transfer_edge_smoothing if text_masked_transfer_edge_smoothing else 5
                if not content_mask_prompt:
                    # Ensure prompts are complete
                    return None
                
                # Get mask
                mask = mask_extractor.perform_mask_extraction(image_filepath, content_mask_prompt)
                # Get overall transfer image from ORIGINAL image
                if depth_style_transfer == d_check_box:
                    depth_image = depth_style.style_Dept(image, input_style)
                if depth_multi_plane == d_check_box:
                    depth_image, _ = depth_style.style_MIP(image, input_style, depth_mip_n)
                # Merge original image with pixel art image
                output_image = segmentation_style_transfer(image, depth_image, mask, edge_smoothing=edge_smoothing)
            elif list_of_text_effects[2] in text_checkbox_values and list_of_text_effects[0] not in text_checkbox_values and list_of_text_effects[1] not in text_checkbox_values:
                # Want to do text based texture masking without location
                style_mask_prompt = text_style_masking_box
                blur_strength = text_emoji_blur_strength if text_emoji_blur_strength else 95
                step_size_multiplier = text_emoji_step_size if text_emoji_step_size else 0.5
                style_strength = text_masked_style_strength if text_masked_style_strength else 1.5
                if not style_mask_prompt:
                    # Ensure prompts are complete
                    return None

                # Get overall transfer image from ORIGINAL image
                if depth_style_transfer == d_check_box:
                    depth_image = depth_style.style_Dept(image, input_style)
                if depth_multi_plane == d_check_box:
                    depth_image, _ = depth_style.style_MIP(image, input_style, depth_mip_n)

                # Get texture mask
                emoji_mask = emoji_mask_extractor.perform_emoji_mask_extraction(style_mask_prompt)
                mask = np.ones_like(style_image)[:, :, 0]
                output_image = emoji_segmentation_style_transfer(image, depth_image, mask, emoji_mask,
                                                                blur_strength=blur_strength,
                                                                step_size_multiplier=step_size_multiplier,
                                                                style_strength=style_strength)
            elif list_of_text_effects[2] in text_checkbox_values and list_of_text_effects[1] in text_checkbox_values and list_of_text_effects[0] not in text_checkbox_values:
                # Want to do text based location masking with texture!
                content_mask_prompt = text_location_box
                style_mask_prompt = text_style_masking_box
                blur_strength = text_emoji_blur_strength if text_emoji_blur_strength else 95
                step_size_multiplier = text_emoji_step_size if text_emoji_step_size else 0.5
                style_strength = text_masked_style_strength if text_masked_style_strength else 1.5
                if not content_mask_prompt or not style_mask_prompt:
                    # Ensure prompts are complete
                    return None
                
                # Get overall transfer image from ORIGINAL image
                if depth_style_transfer == d_check_box:
                    depth_image = depth_style.style_Dept(image, input_style)
                if depth_multi_plane == d_check_box:
                    depth_image, _ = depth_style.style_MIP(image, input_style, depth_mip_n)

                mask = mask_extractor.perform_mask_extraction(image_filepath, content_mask_prompt)
                emoji_mask = emoji_mask_extractor.perform_emoji_mask_extraction(style_mask_prompt)
                output_image = emoji_segmentation_style_transfer(image, depth_image, mask, emoji_mask,
                                                                blur_strength=blur_strength,
                                                                step_size_multiplier=step_size_multiplier,
                                                                style_strength=style_strength)
            else:
                # Text transfer was done on the image, simply do palette transfer on the completed text image!
                if depth_style_transfer == d_check_box:
                    output_image = depth_style.style_Dept(output_image, input_style)
                if depth_multi_plane == d_check_box:
                    output_image, _ = depth_style.style_MIP(output_image, input_style, depth_mip_n)
        else: # text-based was not selected, do normal depth style transfer on whatever image appears
            if depth_style_transfer == d_check_box:
                output_image = depth_style.style_Dept(output_image, input_style)
            if depth_multi_plane == d_check_box:
                output_image, _ = depth_style.style_MIP(output_image, input_style, depth_mip_n)

    if not output_image:
        return None

    return output_image

def apply_video_process(video_filepath, checkbox_values, slowmo_slider_input=None, interpolation_slider_input=None, input_style=None,
                        text_checkbox_values=None, text_box=None, text_location_box=None, text_style_masking_box=None, text_masked_transfer_edge_smoothing=None, text_emoji_blur_strength=None, text_emoji_step_size=None, text_masked_style_strength=None,
                        p_size_slider=0.4, p_checkbox=list, p_colour_dropbox=0, p_colour_interpolate=False, p_edge_slider=50, p_select_im=False, p_in=None, p_in_slid=10,
                        style_image_weight=None, style_image1=None, style_image2=None, color_palette_style=None,
                        d_check_box=None, depth_mip_n=2):
    """
    Apply the selected image processing effects on a video. Save the video in a temporary directory.

    Args:
        video_filepath (str): Input video filepath.
        checkbox_values (list): Selected main effects.
        slowmo_slider_input (float, optional): Multiplier to adjust the speed of the video
        interpolation_slider_input (int, optional): How many cross-dissolved interpolation frames to put between real frames in video
        input_style (PIL.Image, optional): Style image.
        text_checkbox_values (list, optional): Selected text effects
        text_box (str, optional): Selected text style
        text_location_box (str, optional): Selected text mask
        text_style_masking_box (str, optional): Selected text style mask
        text_masked_transfer_edge_smoothing (int, optional): how much to smooth the edges of the content mask
        text_emoji_blur_strength (int, optional):  how much to allow the style mask to escape the content mask
        text_emoji_step_size (float, optional): how often to apply the style mask
        text_masked_style_strength (float, optional): how strong the style should be applied
        p_size_slider (float, optional): Slider value for pixel size
        p_checkbox (list, optional): Selected pixel art effects
        p_colour_dropbox (int, optional): Selected colour palette
        p_colour_interpolate (bool, optional): Interpolate colours
        p_edge_slider (int, optional): Edge threshold
    Returns:
        output_video_filepath (str): Processed output video filepath
    """
    # If no input video, return an empty video
    if not video_filepath:
        return None

    # Open the input video
    cap = cv2.VideoCapture(video_filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)  # get fps of original image
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # get number of frames in original image
    frames = []  # to store list of processed np array frames to be written to the video (in cv2 BGR format)
    # Temporary directory to store image frames (because apply_image_process requires image filepaths)
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_count = 0
        while cap.isOpened():  # while video is open
            ret, frame = cap.read()  # get frame
            if not ret:  # if frame unsuccessfully read, video is no longer playing so break
                break

            # Convert the frame to image file and save temporarily
            frame_filename = os.path.join(temp_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)

            # Apply image processing to each frame
            processed_frame = apply_image_process(frame_filename, checkbox_values, input_style,
                                                  text_checkbox_values, text_box, text_location_box, text_style_masking_box, text_masked_transfer_edge_smoothing, text_emoji_blur_strength, text_emoji_step_size, text_masked_style_strength, p_size_slider, p_checkbox,
                                                  p_colour_dropbox, p_colour_interpolate, p_edge_slider, p_select_im, p_in, p_in_slid,
                                                  style_image_weight, style_image1, style_image2, color_palette_style,
                                                  d_check_box, depth_mip_n)

            # Convert PIL image to numpy array BGR format for video saving later
            if processed_frame.mode == 'RGB':
                # Convert RGB PIL to BGR for OpenCV
                processed_frame = cv2.cvtColor(np.array(processed_frame), cv2.COLOR_RGB2BGR)
            elif processed_frame.mode == 'L':  # grayscale
                # Convert grayscale PIL to BGR for OpenCV
                processed_frame = cv2.cvtColor(np.array(processed_frame), cv2.COLOR_GRAY2BGR)
            else:
                print("WARNING: Returned processed image is in weird non-PIL format...")

            # Append frame
            frames.append(processed_frame)

            print(f"Finished processing frame: {frame_count + 1} out of {num_frames}")  # Status message

            frame_count += 1

        cap.release()  # Release the input video

    # If frames were processed, create the video
    if len(frames) > 0:
        number_of_interpolations = interpolation_slider_input  # how many interpolation frames to put between real frames
        if number_of_interpolations:
            # Use cross-dissolved interpolation frames to smooth differences between frames
            final_frames = [frames[0]]  # handle single frame case
            if len(frames) > 1:
                for frame in frames[1:]:  # for every subsequent frame
                    # Calculate and add the interpolated frames
                    prev_frame = final_frames[-1]
                    for i in range(number_of_interpolations):
                        # Calculate how much to weigh new frame
                        alpha = (i + 1) / (
                                number_of_interpolations + 1)  # = 0.25, 0.5, 0.75 for #=3; 0.33, 0.67 for #=2

                        # Create the interpolated frame and add it to the frame list
                        interpolated = cv2.addWeighted(prev_frame, 1 - alpha, frame, alpha, 0)
                        final_frames.append(interpolated)
                    # Add the actual next frame to the list
                    final_frames.append(frame)
        else:
            final_frames = frames

        # Define temporary output video filepath
        output_video_filepath = os.path.join(video_temp_dir.name, "output_video.mp4")

        # Define output video format and codec
        f_height, f_width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # standard browser codec with mp4 files
        # Set the framerate correctly according to the number of interpolation frames!
        new_fps = fps if not number_of_interpolations else fps * (number_of_interpolations + 1)
        if slowmo_slider_input:
            new_fps = math.floor(new_fps * slowmo_slider_input)
        out = cv2.VideoWriter(output_video_filepath, fourcc, new_fps, (f_width, f_height))

        # Write frames in video
        for frame in final_frames:
            out.write(frame)

        # Release output video
        out.release()

        print(f"Current output video temporarily saved at: {output_video_filepath}")
        return output_video_filepath
    else:  # if no frames returned, return empty video filepath
        return None



# ======================
# 3. UI CONFIGURATION
# ======================

css = """
#img-display-container {
    max-height: 100vh;
}
#img-display-input {
    max-height: 80vh;
}
#img-display-output {
    max-height: 80vh;
}
#download {
    height: 62px;
}
"""

title = "Image Processing Demo"
description = "Apply various style effect, Group 16"
with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    # Choice Row: Does the user want to process a video or an image
    with gr.Row():
        input_type = gr.Radio(choices=["Image", "Video"], label="Select Input Type", value="Image")

    # Input Row: Image/Video input and main effect selection
    with gr.Row():
        image_input = gr.Image(label="Input Image", type="filepath")
        video_input = gr.Video(label="Input Video", visible=False)
        checkbox_input = gr.CheckboxGroup(choices=list_of_effects, label="Apply Effects")

    # Effect-specific components (hidden by default)
    with gr.Row(visible=False) as THIS_ROW:
        input_style = gr.Image(label="Input Style", type="pil", visible=False)

    # Text Effect specific components (hidden by default, only shown when "Text-Based Effects" is selected in main effects)
    with gr.Row(visible=False) as TEXT_ROW:
        text_specific_style_checkbox = gr.CheckboxGroup(choices=list_of_text_effects,
                                                        label="Choose your text-specific effects", visible=False)
        text_style_transfer_text_box = gr.Textbox(lines=1, label="Enter what style you want:", placeholder="fire",
                                                  visible=False)
        location_masking_text_box = gr.Textbox(lines=1, label="Enter what in your input image you want to process:",
                                               visible=False, placeholder="boat")
        style_masking_text_box = gr.Textbox(lines=1, label="Enter what style you want to mask:", visible=False,
                                            placeholder="fire")

    with gr.Row(visible=False) as TEXT_OPTIONS_ROW:
        text_masked_transfer_edge_smoothing = gr.Slider(0, 20, value=5, step=1, visible=False,
                                                        label="Mask Edge Smoothing (Slider)")
        text_emoji_blur_strength = gr.Slider(0, 200, value=95, step=5, visible=False,
                                             label="Content Mask Edge Blur Strength (Slider)")
        text_emoji_step_size = gr.Slider(0.1, 2.5, value=0.5, step=0.1, visible=False,
                                         label="Style Mask Step Size (Slider)")
        text_masked_style_strength = gr.Slider(0, 10, step=0.25, value=1.5, visible=False,
                                               label="Masked Style Strength (Slider)")

    # pixel art components (hidden by default, only shown when "Pixel Art" is selected in main effects)
    with gr.Row(visible=False) as PIXEL_ART:
        p_size_slider = gr.Slider(0.01, 1, value=0.4, step=0.02, visible=False, label="Pixel Size Slider")
        p_checkbox = gr.CheckboxGroup(choices=list_pixel_art_effects, label="Pixel Art Effects", visible=False)

    with gr.Row(visible=False) as PIXEL_ART_OPTIONS:
        p_colour_dropbox = gr.Dropdown(choices=[i for i, _ in enumerate(colour_palette_list)], value=0,
                                       label="Colour Palette", visible=False)
        p_select_im = gr.Checkbox(label="Input Image", info="Select a image to extract colour from", visible=False,
                                  value=False)
        p_colour_interpolate = gr.Checkbox(label="Convert Pallet to Gradient",
                                           info="Linearly Interpolates colour make a gradient", visible=False,
                                           value=False)
        p_out = gr.Image(label="Output Image", type="pil", visible=False, width=20, height=100)

    with gr.Row(visible=False) as PIXEL_ART_COLOUR:
        p_in = gr.Image(label="Input Image", type="pil", visible=False)
        p_in_slid = gr.Slider(0, 20, value=10, step=1, visible=False, label="Number of Colours")

    with gr.Row(visible=False) as PIXEL_ART_EDGE:
        p_edge_slider = gr.Slider(0, 100, value=50, step=1, visible=False, label="Edge Threshold")

    # Video-specific components (hidden by default, only shown when "Video" is selected in input type)
    with gr.Row(visible=False) as VIDEO_OPTIONS_ROW:
        slowmo_slider_input = gr.Slider(0.1, 2, value=1, step=0.1, label="Speed Multiplier (Slider)")
        interpolation_slider_input = gr.Slider(0, 5, step=1, label="Number of Interpolation Frames to Insert (Slider)")

    with gr.Row(visible=False) as STYLE_MIXING_ROW:
        style_image1 = gr.Image(label=f"Style Image 1", type="filepath", visible=False)
        style_image2 = gr.Image(label=f"Style Image 2", type="filepath", visible=False)
        style_image_weight = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.5, visible=False, label="Style Image Weight")

    with gr.Row(visible=False) as STYLE_MIXING_CHANNEL_ATT:
        channel_att = gr.Checkbox(label="Enhance Content", info="Applies channel attention to the content image for enhanced appearance",
                                  visible=False, value=False)

    with gr.Row(visible=False) as COLOR_PALETTE_ROW:
        color_palette_style = gr.Image(label="Input Color Palette", type="filepath", visible=False)

    with gr.Row(visible=False) as DEPTH_ROW:
        d_checkbox = gr.Radio(choices=list_of_depth_effects, value=depth_style_transfer, label="Choose your depth effects", visible=False)

    with gr.Row(visible=False) as DEPTH_MIP_ROW:
        depth_mip_n = gr.Slider(2, 10, value=2, step=1, visible=False, label="Number of Planes")
        d_out = gr.Gallery(label="Output Image", type="pil", visible=False, columns=[3], rows=[1], object_fit="contain", height="auto")

    # Start process row for images
    with gr.Row() as ADMIN_ROW:
        start_button = gr.Button("Start Processing Image")

    # Start process row for videos (default: hidden)
    with gr.Row(visible=False) as VIDEO_ADMIN_ROW:
        video_start_button = gr.Button("Start Processing Video")


# ======================
# 4. VISIBILITY HANDLERS
# ======================

    def update_style_visibility(checkbox_values):
        """
        Update the visibility of components based on selected effects.
        #TODO: Add new effect components here.
        """
        component_mapping = [
            (style, gr.Row),
            (style, gr.Image),
            ([text], gr.CheckboxGroup),
            ([text], gr.Row),
            ([text], gr.Row),
            ([pixel], gr.Row),
            ([pixel], gr.Row),
            ([pixel], gr.Row),
            ([pixel], gr.Row),
            ([pixel], gr.Slider),
            ([pixel], gr.CheckboxGroup),
            ([mixed_style], gr.Row),
            ([mixed_style], gr.Number),
            ([mixed_style], gr.Image),
            ([mixed_style], gr.Image),
            ([mixed_style], gr.Row),
            ([mixed_style], gr.Checkbox),
            ([color_palette], gr.Row),
            ([color_palette], gr.Image),
            ([depth], gr.Row),
            ([depth], gr.Row),
            ([depth], gr.Radio),
        ]
        re_set = [
            component_class(visible=(any(elem in checkbox_values for elem in condition_list)))
            for condition_list, component_class in component_mapping
        ]
        return re_set


    # Trigger visibility updates based on selected effects
    checkbox_input.change(
        fn=update_style_visibility,
        inputs=checkbox_input,
        outputs=[THIS_ROW, input_style, text_specific_style_checkbox, TEXT_ROW, TEXT_OPTIONS_ROW, PIXEL_ART, PIXEL_ART_OPTIONS, PIXEL_ART_COLOUR, PIXEL_ART_EDGE, p_size_slider,
                 p_checkbox, STYLE_MIXING_ROW, style_image_weight, style_image1, style_image2, STYLE_MIXING_CHANNEL_ATT, channel_att, COLOR_PALETTE_ROW, color_palette_style, DEPTH_ROW, DEPTH_MIP_ROW, d_checkbox],
    )


    def update_text_style_visibility(checkbox_values):
        """
        Update the visibility of text related components based on selected effects.
        """
        component_mapping = [
            (text_style_transfer, gr.Textbox),
            (text_location_masking, gr.Textbox),
            (text_style_masking, gr.Textbox),
        ]
        re_set = [
            component_class(visible=(condition_list in checkbox_values))
            for condition_list, component_class in component_mapping
        ]

        # Handle custom sliders for specific text settings
        re_set += [
            gr.Slider(visible=(True if text_style_transfer in checkbox_values and text_location_masking in checkbox_values and text_style_masking not in checkbox_values else False)),
            gr.Slider(visible=(True if text_location_masking in checkbox_values and text_style_masking in checkbox_values else False)),
            gr.Slider(visible=(True if (text_location_masking in checkbox_values or text_style_transfer in checkbox_values) and text_style_masking in checkbox_values else False)),
            gr.Slider(visible=(True if text_style_transfer in checkbox_values and text_style_masking in checkbox_values else False)),
        ]

        return re_set


    # Trigger visibility updates based on selected effects
    text_specific_style_checkbox.change(
        fn=update_text_style_visibility,
        inputs=text_specific_style_checkbox,
        outputs=[text_style_transfer_text_box, location_masking_text_box, style_masking_text_box,
                 text_masked_transfer_edge_smoothing, text_emoji_blur_strength, text_emoji_step_size,
                 text_masked_style_strength],
    )


    def update_pixel_art_visibility(checkbox_values):
        """
        Update the visibility of components based on selected effects.
        """
        component_mapping = [
            (pixel_colour, gr.Row),
            (pixel_colour, gr.Row),
            (pixel_edge, gr.Row),
            (pixel_colour, gr.Dropdown),
            (pixel_colour, gr.Checkbox),
            (pixel_colour, gr.Image),
            (pixel_colour, gr.Checkbox),
            (pixel_edge, gr.Slider)

        ]

        re_set = [
            component_class(visible=(condition_list in checkbox_values))
            for condition_list, component_class in component_mapping
        ]


        return re_set


    p_checkbox.change(
        fn=update_pixel_art_visibility,
        inputs=p_checkbox,
        outputs=[PIXEL_ART_OPTIONS, PIXEL_ART_COLOUR ,PIXEL_ART_EDGE, p_colour_dropbox, p_colour_interpolate, p_out, p_select_im, p_edge_slider],
    )


    def update_colour_palette_visibility_override(p_select_im):
        if p_select_im:
            return [gr.Image(visible=True), gr.Slider(visible=True), gr.Dropdown(interactive=False)]
        else:
            return [gr.Image(visible=False), gr.Slider(visible=False), gr.Dropdown(interactive=True)]


    p_select_im.change(
        fn=update_colour_palette_visibility_override,
        inputs=p_select_im,
        outputs=[p_in, p_in_slid, p_colour_dropbox],
    )


    def update_colour_palette_visibility(p_colour_dropbox, p_colour_interpolate, p_select_im, p_in, p_in_slid):

        """
        Update the visibility of components based on selected effects.
        """
        if p_select_im and p_in:
            colour_palette.set_palette_from_image(p_in, p_in_slid)
            return colour_palette.display_palette(size, interpolate=p_colour_interpolate)
        else:
            if p_colour_dropbox == None:
                return Image.new('RGB', (100, 100))

            if p_colour_interpolate:
                return colour_palette_list_interpolate[p_colour_dropbox]
            else:
                return colour_palette_list[p_colour_dropbox]


    p_colour_dropbox.change(
        fn=update_colour_palette_visibility,
        inputs=[p_colour_dropbox, p_colour_interpolate, p_select_im, p_in, p_in_slid],
        outputs=p_out,
    )

    p_colour_interpolate.change(
        fn=update_colour_palette_visibility,
        inputs=[p_colour_dropbox, p_colour_interpolate, p_select_im, p_in, p_in_slid],
        outputs=p_out,
    )

    p_in.change(
        fn=update_colour_palette_visibility,
        inputs=[p_colour_dropbox, p_colour_interpolate, p_select_im, p_in, p_in_slid],
        outputs=p_out,
    )

    p_in_slid.change(
        fn=update_colour_palette_visibility,
        inputs=[p_colour_dropbox, p_colour_interpolate, p_select_im, p_in, p_in_slid],
        outputs=p_out,
    )


    def update_depth_visibility(d_checkbox):
        """
        Update the visibility of components based on selected effects.
        """
        component_mapping = [
            (depth_multi_plane, gr.Row),
            (depth_multi_plane, gr.Slider),
            (depth_multi_plane, gr.Gallery),
        ]
        re_set = [
            component_class(visible=(condition_list in d_checkbox))
            for condition_list, component_class in component_mapping
        ]

        return re_set

    d_checkbox.change(
        fn=update_depth_visibility,
        inputs=d_checkbox,
        outputs=[DEPTH_MIP_ROW, depth_mip_n, d_out],
    )

    def update_d_out(depth_mip_n, image_input):
        if image_input:
            return depth_style.depth_split(Image.open(image_input), depth_mip_n)
        else:
            return None

    depth_mip_n.change(
        fn=update_d_out,
        inputs=[depth_mip_n, image_input],
        outputs=[d_out]
    )


    # Output Row: Display processed image
    output_image = gr.Image(label="Output Image", type="pil")
    stylized_images = gr.Gallery(label="Stylized Images", type="pil", visible=False)
    start_button.click(
        fn=apply_image_process,
        inputs=[image_input, checkbox_input, input_style, text_specific_style_checkbox,
                text_style_transfer_text_box, location_masking_text_box, style_masking_text_box,
                text_masked_transfer_edge_smoothing, text_emoji_blur_strength, text_emoji_step_size,
                text_masked_style_strength, p_size_slider, p_checkbox, p_colour_dropbox, p_colour_interpolate,
                p_edge_slider, p_select_im, p_in, p_in_slid, style_image_weight, style_image1, style_image2,
                color_palette_style, d_checkbox, depth_mip_n],
        outputs=output_image
    )

    # Output Row: Display processed video (hidden by default)
    output_video = gr.Video(label="Output Video", visible=False, format='mp4')
    video_start_button.click(
        fn=apply_video_process,
        inputs=[video_input, checkbox_input, slowmo_slider_input, interpolation_slider_input, input_style,
                text_specific_style_checkbox, text_style_transfer_text_box, location_masking_text_box,
                style_masking_text_box, text_masked_transfer_edge_smoothing, text_emoji_blur_strength,
                text_emoji_step_size, text_masked_style_strength, p_size_slider, p_checkbox, p_colour_dropbox,
                p_colour_interpolate, p_edge_slider, p_select_im, p_in, p_in_slid, style_image_weight, style_image1,
                style_image2, color_palette_style, d_checkbox, depth_mip_n],
        outputs=output_video
    )

    def update_channel_attention_flag(channel_att):
        global CHANNEL_ATT_ENABLED
        if channel_att:
            CHANNEL_ATT_ENABLED = True
        else:
            CHANNEL_ATT_ENABLED = False

    channel_att.change(
        fn=update_channel_attention_flag,
        inputs=[channel_att]
    )

    # Switch between image and video input visibility
    def toggle_input_type(selected_type):
        """
        Update the visibility of input related components based on selected input type.
        """
        if selected_type == "Image":
            return [gr.Image(visible=True), gr.Video(visible=False), gr.Row(visible=False), gr.Row(visible=True),
                    gr.Row(visible=False), gr.Image(visible=True), gr.Video(visible=False)]
        else:
            return [gr.Image(visible=False), gr.Video(visible=True), gr.Row(visible=True), gr.Row(visible=False),
                    gr.Row(visible=True), gr.Image(visible=False), gr.Video(visible=True)]


    # Trigger visibility updates based on selected input type
    input_type.change(
        fn=toggle_input_type,
        inputs=input_type,
        outputs=[image_input, video_input, VIDEO_OPTIONS_ROW, ADMIN_ROW, VIDEO_ADMIN_ROW, output_image, output_video]
    )

if __name__ == "__main__":
    demo.launch(show_error=True, show_api=False)
    video_temp_dir.cleanup()  # clean up temporary directory used for processed videos


