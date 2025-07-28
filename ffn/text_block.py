import torch
import torch.nn as nn
from ffn.moe import Llama4TextMoe

class Llama4TextBlock(nn.Module):
    """
    A single Transformer block containing MoE feedforward logic.
    Expects hidden_states from attention output as input.
    """
    def __init__(self, config):
        super().__init__()
        self.moe = Llama4TextMoe(config)
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states):
        """
        hidden_states: [batch_size, seq_len, hidden_dim]
        """
        print("\n[TextBlock] Input shape:", hidden_states.shape)

        # Step 1: Normalize hidden states (Pre-Norm style)
        normed = self.norm(hidden_states)
        print("[TextBlock] After LayerNorm:", normed.shape)

        # Step 2: Apply MoE block (shared MLP + experts)
        moe_out = self.moe(normed)
        print("[TextBlock] After MoE output shape:", moe_out.shape)

        # Step 3: Residual connection
        output = hidden_states + moe_out
        print("[TextBlock] Output shape (with residual):", output.shape)

        return output
