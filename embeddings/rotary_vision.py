import torch
import torch.nn as nn

class Llama4VisionRotaryEmbedding(nn.Module):
    """
    Generates rotary positional encodings for 2D vision inputs.
    This uses X and Y positional components separately and combines them.
    """
    def __init__(self, config):
        super().__init__()
        num_patches_per_dim = config.image_size // config.patch_size
        num_patches = num_patches_per_dim ** 2 + 1  # +1 for CLS token

        freq_dim = config.hidden_size // config.num_attention_heads // 2
        theta = config.rope_theta
        rope_freq = 1.0 / (theta ** (torch.arange(0, freq_dim, 2).float() / freq_dim))

        # Patch (x, y) positions
        patch_indices = torch.arange(num_patches - 1).unsqueeze(1)
        x_coord = patch_indices % num_patches_per_dim
        y_coord = patch_indices // num_patches_per_dim

        # Compute sinusoidal position encodings for x and y
        freqs_x = ((x_coord + 1) * rope_freq).repeat_interleave(2, dim=-1)
        freqs_y = ((y_coord + 1) * rope_freq).repeat_interleave(2, dim=-1)

        # Merge them: interleave to combine x/y rotary info
        freqs = torch.cat([freqs_x, freqs_y], dim=-1)[..., ::2]
        cls_freq = torch.zeros((1, freqs.shape[1]))  # CLS token freq is 0
        freqs = torch.cat([cls_freq, freqs], dim=0)

        # Final shape: (num_patches, head_dim) as complex
        self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs).to(torch.cfloat)

    def forward(self, x):
        return self.freqs_cis.to(x.device)  # (Num Patches, D/2) complex


def vision_apply_rotary_emb(query, key, freqs_ci):
    """
    Applies 2D rotary positional embeddings to the query and key tensors.
    Assumes input is shaped as (batch, seq_len, num_heads, head_dim).
    
    Args:
        query (Tensor): Query tensor of shape [B, Seq, H, D]
        key (Tensor): Key tensor of shape [B, Seq, H, D]
        freqs_ci (Tensor): Complex rotary frequencies of shape [Seq, D/2] or [1, Seq, D/2]

    Returns:
        Tuple[Tensor, Tensor]: (rotated_query, rotated_key)
    """
    # Convert last dim into complex by pairing adjacent dimensions
    query_complex = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
    key_complex = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))

    # Unsqueeze freqs if necessary to match batch/head dims
    freqs_ci = freqs_ci.unsqueeze(0)  # (1, Seq, D/2)
    
    # Element-wise complex multiplication: rotate
    rotated_query = torch.view_as_real(query_complex * freqs_ci).flatten(3)
    rotated_key = torch.view_as_real(key_complex * freqs_ci).flatten(3)

    return rotated_query.type_as(query), rotated_key.type_as(key)
