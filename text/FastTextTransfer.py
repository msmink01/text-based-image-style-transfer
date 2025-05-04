import clip
import numpy as np
import torch
import torchvision.transforms as transforms

from text.subnetworks.ghiasi_img_transformer import GhiasiImgTransformer
from text.subnetworks.clip_text_embedding_transformer import ClipTextEmbeddingTransformer

class FastTextStyleTransfer():
    def __init__(self, device):
        """
        Create a FastTextStyleTransfer object by creating the underlying subnetworks

        Inpired by: https://doi.org/10.48550/arXiv.2210.03461

        Args:
            device (str): device to use for the subnetworks: 'cuda' or 'cpu'
        """
        self.device = device

        self.image_transforms = transforms.Compose([transforms.ToTensor()])

        self.img_transformer = GhiasiImgTransformer()
        self.img_transformer.to(device)
        self.img_transformer.requires_grad_(False)

        self.embedding_transformer = ClipTextEmbeddingTransformer()
        self.embedding_transformer.to(device)
        self.embedding_transformer.requires_grad_(True)

        # Load pretrained clip embedder
        self.clip_text_embedder, _ = clip.load('ViT-B/32', device, jit=False)
        self.clip_text_embedder.to(device)
        self.clip_text_embedder.requires_grad_(False)

    def perform_transfer(self, content_image, text):
        """
        Given an input image path and a text prompt apply the text style to the image

        Args:
            content_image (Image): PIL input image
            text (str): style text prompt

        Returns:
            pil_image (Image): PIL image with applied style
        """
        # transform image to tensor
        content_image_tensor = self.image_transforms(np.array(content_image))[:3, :, :].to(self.device)

        with torch.no_grad():
            # create text embedding
            tokenized_text = clip.tokenize([text]).to(self.device)
            text_embeddings = self.clip_text_embedder.encode_text(tokenized_text).detach()
            text_embeddings = text_embeddings.mean(axis=0, keepdim=True)
            text_embeddings = text_embeddings.type(torch.float32)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

            # transform text embedding into style embedding
            style_embedding = self.embedding_transformer(text_embeddings)
            # create style transferred image with content image and style embedding
            new_image = self.img_transformer(content_image_tensor, style_embedding)
        
        # transform style transferred image into pil image
        pil_image = transforms.functional.to_pil_image(new_image.squeeze())
        
        return pil_image