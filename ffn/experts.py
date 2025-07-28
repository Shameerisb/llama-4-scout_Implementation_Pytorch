import torch
import torch.nn as nn

class Llama4TextExperts(nn.Module):
    """
    This module holds all the local experts for the MoE (Mixture of Experts) mechanism.
    Each expert is a gated feedforward MLP that transforms the token representation.
    """
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = config.intermediate_size

        # Combined gate and up projection: maps from hidden_size to 2 * expert_dim (for gated activation)
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))

        # Down projection to project expert output back to hidden_size
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.expert_dim, self.hidden_size))

        # Activation used in the gate
        self.act_fn = nn.SiLU()

        # Initialize expert weights
        nn.init.normal_(self.gate_up_proj)
        nn.init.normal_(self.down_proj)

    def forward(self, hidden_states):
        # Expecting (num_experts * tokens_per_expert, hidden_size)
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)

        # Apply linear projection to get gate and up branches
        gate_up = torch.bmm(hidden_states, self.gate_up_proj)  # (num_experts, tokens, 2 * expert_dim)
        gate, up = gate_up.chunk(2, dim=-1)

        # Apply gated activation: SiLU(gate) * up
        gated = up * self.act_fn(gate)

        # Project output back to hidden_size
        output = torch.bmm(gated, self.down_proj)

        # Flatten back to (num_experts * tokens, hidden_size)
        return output.view(-1, self.hidden_size)
