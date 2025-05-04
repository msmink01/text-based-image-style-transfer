import numpy as np
import torch
import cv2
from PIL import Image
from fontTools.subset import prune_post_subset

from text.subnetworks.groundingdino_text_object_detector import GroundingDINOTextObjectDetector
from text.subnetworks.segment_anything_object_segmenter import SegmentAnythingObjectSegmenter

class TextMaskExtractor():
    def __init__(self, device='cuda'):
        """
        Create a TextMaskExtractor object by creating the underlying subnetworks

        Args:
            device (str): device to use for the subnetworks: 'cuda' or 'cpu'
        """
        self.device = device

        print("Loading Text Object Detector...")
        self.text_object_detector = GroundingDINOTextObjectDetector(device=device)
        print("Loading Object Segmenter...")
        self.object_segmenter = SegmentAnythingObjectSegmenter(device=device)

    def perform_mask_extraction(self, image_path, text_prompt):
        """
        Given an input image path and a text prompt extract a singular mask for a text prompt

        Args:
            image_path (str): path to the input image
            text_prompt (str): object to extract mask for (Example: "boat")

        Returns:
            mask (numpy array): True/False mask of where in image object appears
        """
        # Preprocess image for dino
        image_pil = Image.open(image_path).convert("RGB")  # load image
        image_pil = self._preprocess_image(image_pil)
        
        # Get filtered bounding boxes for text prompt
        boxes_filt, pred_phrases = self.text_object_detector(image_pil, text_prompt)

        # Preprocess image for segmentation
        image = cv2.imread(image_path)
        image = np.asarray(self._preprocess_image(image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Transform bounding box coordinates to be in cv2 format (not 0-1)
        size = image_pil.size
        H, W = size[1], size[0]

        if not boxes_filt.size(0): # if the number of filtered boxes is 0, return an empty mask
            return np.full((H, W), False)
        
        for i in range(boxes_filt.size(0)): # transform every bbox to (0 - H or W) range
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        boxes_filt = boxes_filt.cpu()

        # Get segmentation masks
        masks = self.object_segmenter(image, boxes_filt)
        # Combine separate masks into one mask
        masks = torch.sum(masks, dim=0).unsqueeze(0)
        masks = torch.where(masks > 0, True, False)
        mask = masks[0][0].cpu().numpy()

        return mask

    def _preprocess_image(self, image, resize=False, square=False, height=512, width=512, left=0, right=0, top=0, bottom=0):
        """
        Internal function to preprocess an image to be used for mask extraction.

        Will crop off defined parts of image before resizing to a specified shape.

        Args:
            image (str or Image or cv2 image): PIL/cv2 image or path to image to be preprocessed
            resize (bool): whether to resize the image at the end or not
            square (bool): whether to crop the image to be square or not
            height (int): height to resize cropped image to (if resize=True)
            width (int): width to resize cropped image to (if resize=True)
            left (int): how much of image to cut off from left before resizing
            right (int): how much of image to cut off from right before resizing
            top (int): how much of image to cut off from top before resizing
            bottom (int): how much of image to cut off from bottom before resizing

        Returns: 
            image (Image): PIL image cropped and resized as specified
        """
        # get image as np array
        if isinstance(image, str):
            image = np.array(Image.open(image))
        elif isinstance(image, np.ndarray):
            pass
        else:
            image = np.array(image)
            
        # get height/width of current image
        if image.ndim == 3:
            image = image[:, :, :3]
            h, w, _ = image.shape
        else:
            h, w = image.shape
            
        # crop image according to args
        left = min(left, w-1)
        right = min(right, w - left - 1)
        top = min(top, h - left - 1)
        bottom = min(bottom, h - top - 1)
        image = image[top:h-bottom, left:w-right]
        
        # get updated width/height of cropped image
        if image.ndim == 3:
            h, w, _ = image.shape
        else:
            h, w = image.shape

        # crop the image to be square
        if square:
            if h < w:
                offset = (w - h) // 2
                image = image[:, offset:offset + h]
            elif w < h:
                offset = (h - w) // 2
                image = image[offset:offset + w]

        # convert np array into PIL image and resize
        image = Image.fromarray(image)
        if resize:
            image = image.resize((height, width))

        return image