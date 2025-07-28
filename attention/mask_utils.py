import torch
import torch.nn as nn
import torch.nn.functional as F
from embeddings.rotary_vision import vision_apply_rotary_emb
class Llama4VisionAttention(nn.Module):
    """
    Standard ViT-style attention (with rotary embeddings).
    """
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states, freqs_ci):
        B, L, _ = hidden_states.shape

        # Project and reshape to (B, L, H, D)
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)

        # Apply rotary positional encodings
        q, k = vision_apply_rotary_emb(q, k, freqs_ci)

        # Transpose to (B, H, L, D) for dot-product attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)

        # Flatten back: (B, L, E)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        return self.o_proj(attn_out)
