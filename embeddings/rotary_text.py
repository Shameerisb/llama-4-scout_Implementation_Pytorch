import torch
import torch.nn as nn

class Llama4TextRotaryEmbedding(nn.Module):
    """
    Generates complex sinusoidal frequencies for rotary embeddings in text attention.
    """
    def __init__(self, config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta

        # Inverse frequencies as per RoPE paper
        inv_freq = 1.0 / (self.rope_theta ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device) / self.head_dim
        ))

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        """
        Returns complex rotary embeddings for each token position.
        x: (B, L, H, D)
        position_ids: (B, L)
        """
        B, L = position_ids.shape
        freqs = torch.einsum("i,j->ij", position_ids.float().view(-1), self.inv_freq)
        emb = torch.polar(torch.ones_like(freqs), freqs)  # e^(jθ)
        emb = emb.view(B, L, 1, -1)  # Match attention shape
        return emb  # Complex tensor (B, L, 1, D//2)
