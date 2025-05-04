import torch
from segment_anything import sam_model_registry, SamPredictor 

class SegmentAnythingObjectSegmenter(torch.nn.Module):
    def __init__(
            self, 
            sam_checkpoint = "text/subnetworks/checkpoints/sam_vit_b_01ec64.pth",
            model_type = "vit_b",
            device='cuda',):
        """
        Create a SegmentAnythingObjectSegmenter object by creating the underlying segment anything predictor

        Args:
            device (str): device to use for the subnetworks: 'cuda' or 'cpu'
        """
        super(SegmentAnythingObjectSegmenter, self).__init__()

        self.device = device
        self.model = SamPredictor(sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device))

    def forward(self, image, boxes_filt):
        """
        The forward pass of the SegmentAnythingObjectSegmenter.

        Args:
            image (cv2 image): cv2 RGB image to get the masks from
            boxes_filt (np array): the bounding boxes to get the masks from, value range: 0 to height/width of image

        Return:
            masks (Tensor): tensor of masks for each bbox (N x 1 x H x W)
        """
        # apply image and boxes to model
        self.model.set_image(image)
        transformed_boxes = self.model.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)
        
        # get masks
        masks, _, _ = self.model.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(self.device),
            multimask_output = False,
        )

        return masks