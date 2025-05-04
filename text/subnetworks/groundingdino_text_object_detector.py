import torch
import numpy as np
import groundingdino
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

class GroundingDINOTextObjectDetector(torch.nn.Module):
    def __init__(
            self, 
            model_config_path='text/subnetworks/checkpoints/GroundingDINO_SwinT_OGC.py',
            model_checkpoint_path='text/subnetworks/checkpoints/groundingdino_swint_ogc.pth',
            device='cuda',
            box_threshold = 0.3,
            text_threshold = 0.5):
        """
        Create a GroundingDINOTextObjectDetector object by creating the underlying groundingDINO predictor

        Args:
            model_config_path (str): path the the grounding dino config py file
            model_checkpoint_path (str): path to the grounding dino weights file
            device (str): device to use for the subnetworks: 'cuda' or 'cpu'
            box_threshold (float): threshold when to include detected bboxes
            text_threshold (float): threshold when to include detected bboxes based on text prompt
        """
        super(GroundingDINOTextObjectDetector, self).__init__()

        self.device = device

        args = SLConfig.fromfile(model_config_path)
        args.device = device
        self.model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        self.model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.model.eval()
        self.model.to(device)
    
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def forward(self, image_pil, det_prompt):
        """
        The forward pass of the GroundingDINOTextObjectDetector.
        Will get the bboxes in the image based on the text prompt and filter out any bboxes not passing our thresholds.

        Args:
            image_pil (Image): PIL image to detect the objects from
            det_prompt (str): the prompt to detect in the image

        Return:
            boxes_filt (Tensor): tensor of the detected bboxes (N x 1 x H x W)
            pred_phrases (List[str]): list of the prediction phrases for each bbox (Example: ['hat(0.53)', ...])
        """
        # transform PIL image to tensor
        image, _ = self.transform(image_pil, None)  # 3, h, w
        image = image.to(self.device)

        # preprocess text prompt
        if not det_prompt.endswith('.'):
            det_prompt += '.'

        # get 900 predicted bboxes for the prompt
        with torch.no_grad():
            outputs = self.model(image[None], captions=[det_prompt])
        
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

        # filter output by box threshold
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # filter output by text threshold        
        tokenlizer = self.model.tokenizer
        tokenized = tokenlizer(det_prompt) # tokenize the prompt
        pred_phrases = []
        logits_filt_item = []
        filt_mask = [] # what to filter the bboxes by after this process
        for logit in logits_filt:
            pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenlizer) # get whether prompt matches bbox based on text thresh
            if pred_phrase: # if the confidence in the box was high enough to be labeled as the phrase
                logits_filt_item.append(logit.max().item())
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
                filt_mask.append(1)
            else: # this box likely does not containt his phrase
                filt_mask.append(0)

        # filter bboxes by text threshold
        filt_mask = np.where(np.array(filt_mask) > 0, True, False)
        boxes_filt = boxes_filt[filt_mask]

        return boxes_filt, pred_phrases