import torch
import torch.nn as nn
from .unfold_conv import Llama4UnfoldConvolution
from .vision_encoder import Llama4VisionEncoder
from .pixel_shuffle_mlp import Llama4VisionPixelShuffleMLP
from embeddings.rotary_vision import Llama4VisionRotaryEmbedding

class Llama4VisionModel(nn.Module):
    """
    End-to-end vision transformer for LLaMA 4: converts images to patch embeddings,
    applies rotary attention, encodes with ViT layers, and adapts for text fusion.
    """
    def __init__(self, config):
        super().__init__()
        self.patch_embedding = Llama4UnfoldConvolution(config)
        self.encoder = Llama4VisionEncoder(config)
        self.pixel_shuffle_proj = Llama4VisionPixelShuffleMLP(config)
        self.rotary = Llama4VisionRotaryEmbedding(config)

        self.class_token = nn.Parameter(torch.randn(config.hidden_size))
        self.positional_embedding = nn.Parameter(torch.randn((config.image_size // config.patch_size)**2 + 1, config.hidden_size))
        self.norm_pre = nn.LayerNorm(config.hidden_size)
        self.norm_post = nn.LayerNorm(config.hidden_size)

    def forward(self, images):
        B = images.size(0)
        x = self.patch_embedding(images)              # (B, Num_Patches, Hidden_Dim)
        cls = self.class_token.unsqueeze(0).expand(B, 1, -1)
        x = torch.cat([x, cls], dim=1)                # Add CLS token

        x = x + self.positional_embedding             # Add position encoding
        x = self.norm_pre(x)

        freqs = self.rotary(x)
        x = self.encoder(x, freqs)
        x = self.norm_post(x)

        x = x[:, :-1]  # remove CLS
        return self.pixel_shuffle_proj(x)
