import torch.nn as nn
from attention.vision_attention import Llama4VisionAttention
from ffn.mlp import Llama4VisionMLP  # Reuse standard MLP

class Llama4VisionEncoderLayer(nn.Module):
    """
    A single ViT-style encoder block with attention + feedforward.
    """
    def __init__(self, config):
        super().__init__()
        self.attn = Llama4VisionAttention(config)
        self.mlp = Llama4VisionMLP(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x, freqs_ci):
        # Pre-norm + Attention
        x = x + self.attn(self.norm1(x), freqs_ci)
        # Pre-norm + MLP
        x = x + self.mlp(self.norm2(x))
        return x

class Llama4VisionEncoder(nn.Module):
    """
    Stack of vision transformer layers.
    """
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([Llama4VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x, freqs_ci):
        for layer in self.layers:
            x = layer(x, freqs_ci)
        return x
