import torch
import torch.nn as nn
import torch.nn.functional as F
from norm.l2_norm import Llama4TextL2Norm
from embeddings.utils import apply_rotary_emb

def repeat_kv(x, n_rep):
    """
    Repeats K/V heads to match the number of Q heads for GQA.
    Shape: (B, H, L, D) -> (B, H * n_rep, L, D)
    """
    B, H, L, D = x.shape
    x = x[:, :, None, :, :].expand(B, H, n_rep, L, D)
    return x.reshape(B, H * n_rep, L, D)

class Llama4TextAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.use_rope = (layer_idx + 1) % config.no_rope_layer_interval == 0

        # Linear projections for QKV
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # Norm used before dot-product attention
        self.qk_norm = Llama4TextL2Norm() if config.use_qk_norm and self.use_rope else None

        # Scaling
        self.scaling = self.head_dim ** -0.5

    def forward(self, hidden_states, freqs_cis, attention_mask, past_key_value, cache_position):
        """
        freqs_cis: rotary complex values
        past_key_value: Cache object
        """
        B, L, _ = hidden_states.shape
        H = self.num_heads
        D = self.head_dim

        # Project Q, K, V
        q = self.q_proj(hidden_states).view(B, L, H, D)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, D)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, D)

        # Rotary position embeddings (only applied to certain layers)
        if self.use_rope:
            q, k = apply_rotary_emb(q, k, freqs_cis.to(q.device))

        if self.qk_norm is not None:
            q = self.qk_norm(q)
            k = self.qk_norm(k)

        # Transpose for attention: (B, H, L, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Update and retrieve cached K/V
        if past_key_value is not None:
            k, v = past_key_value.update(k, v, self.layer_idx)

        # Repeat KV heads to match query head count
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        # Scaled dot-product attention with masking
        attn_scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores.float(), dim=-1).type_as(q)
        attn_out = torch.matmul(attn_probs, v)

        # Combine heads and project back
        attn_out = attn_out.transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(attn_out)
