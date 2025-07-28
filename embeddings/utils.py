import torch

def apply_rotary_emb(q, k, freqs_cis):
    """
    Apply complex rotary embeddings to Q and K tensors.
    Expected shape: q/k: (B, L, H, D), freqs_cis: (B, L, 1, D/2) complex
    """
    # Split real/imag pairs for complex rotation
    def to_complex(x):
        return torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    q_ = to_complex(q)
    k_ = to_complex(k)

    q_rot = q_ * freqs_cis
    k_rot = k_ * freqs_cis

    # Convert back to real and flatten
    q_rot = torch.view_as_real(q_rot).flatten(-2)
    k_rot = torch.view_as_real(k_rot).flatten(-2)

    return q_rot.type_as(q), k_rot.type_as(k)
