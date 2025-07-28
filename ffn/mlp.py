import torch.nn as nn
import torch

class Llama4TextMLP(nn.Module):
    """
    Shared dense MLP used in parallel with MoE.
    Applies a gated feedforward network to all tokens.
    """
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        # Gated activation path: SiLU(gate_proj(x)) * up_proj(x)
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)

        # Project back to hidden size
        return self.down_proj(gated)
