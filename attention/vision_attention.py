import torch
import torch.nn as nn
import torch.nn.functional as F
from embeddings.rotary_vision import vision_apply_rotary_emb

class Llama4VisionAttention(nn.Module):
    """
    Vision Transformer-style multi-head self-attention with 2D rotary embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = config.attention_dropout

        # QKV projections
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        # Final projection after attention
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(self, hidden_states, freqs_ci):
        """
        Args:
            hidden_states: [B, Seq, D]
            freqs_ci:      [Seq, D/2] complex, for rotary position embeddings

        Returns:
            Tensor of shape [B, Seq, D]
        """
        B, S, D = hidden_states.shape
        H = self.num_heads
        HD = self.head_dim

        # Compute Q, K, V and reshape to [B, Seq, H, D/H]
        q = self.q_proj(hidden_states).view(B, S, H, HD)
        k = self.k_proj(hidden_states).view(B, S, H, HD)
        v = self.v_proj(hidden_states).view(B, S, H, HD)

        # Apply rotary embeddings to Q and K
        q, k = vision_apply_rotary_emb(q, k, freqs_ci)

        # Transpose for attention: [B, H, S, D/H]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled Dot Product Attention
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, is_causal=False)

        # Merge heads: [B, S, D]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)

        # Final linear projection
        return self.o_proj(attn_out)
