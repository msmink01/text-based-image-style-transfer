import torch
import torch.nn as nn

class ClipTextEmbeddingTransformer(nn.Module):
    def __init__(self):
        """
        Create a ClipTextEmbeddingTransformer object.

        Is a FeedForward NN with 5 dense layers with LeakyReLU and TanH activations. Transforms CLIP text embeddings into style embeddings.

        Inpired by: https://doi.org/10.48550/arXiv.2210.03461
        """
        super(ClipTextEmbeddingTransformer, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 150),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(150, 150),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(150, 100),
            nn.Tanh(),
        )

        self.load_state_dict(torch.load("text/subnetworks/checkpoints/clip_text_embedding_transformer.pth", map_location=torch.device('cpu')), strict=True)

    def reset_weights(self):
        """
        Reload the weights of the model.
        """
        self.load_state_dict(torch.load("text/subnetworks/checkpoints/clip_text_embedding_transformer.pth", map_location=torch.device('cpu')), strict=True)

    def forward(self, text_embeddings):
        """
        Forward pass of the ClipTextEmbeddingTransformer object.

        Args:
            text_embeddings (Tensor): input tensor of CLIP text embeddings

        Output:
            latent_space_sample (Tensor): tensor of style embeddings
        """
        latent_space_sample = self.model(text_embeddings)
        return latent_space_sample