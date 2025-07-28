import torch
import torch.nn as nn
from .experts import Llama4TextExperts
from .mlp import Llama4TextMLP

class Llama4TextMoe(nn.Module):
    """
    Wraps the MoE logic: routes tokens to experts, gathers output, and adds to shared MLP.
    """
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts

        self.experts = Llama4TextExperts(config)
        self.shared_expert = Llama4TextMLP(config)

        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)

        print("Initialized Llama4TextMoe:")
        print(f"  Num experts       : {self.num_experts}")
        print(f"  Experts per token : {self.top_k}")
        print(f"  Hidden dim        : {self.hidden_dim}")

    def forward(self, hidden_states):
        batch_size, seq_len, embed_dim = hidden_states.shape
        print("\n[Llama4TextMoe] Input shape:", hidden_states.shape)

        flat_tokens = hidden_states.view(-1, embed_dim)  # [B * L, D]
        print("[Step 1] Flattened tokens for routing:", flat_tokens.shape)

        # Router scores for each expert
        router_logits = self.router(flat_tokens)  # [B * L, E]
        print("[Step 2] Router logits shape:", router_logits.shape)

        # Top-k expert scores and indices
        top_vals, top_idx = torch.topk(router_logits, self.top_k, dim=-1)
        print("[Step 3] Top-k values shape:", top_vals.shape)
        print("[Step 3] Top-k indices shape:", top_idx.shape)
        print("Example top-2 experts for first token:", top_idx[0].tolist())

        # Full score matrix initialized to -inf and populated with top-k scores
        router_scores = torch.full_like(router_logits, float('-inf')).scatter_(1, top_idx, top_vals)
        router_scores = router_scores.transpose(0, 1)  # [E, B * L]
        router_scores = torch.sigmoid(router_scores.float()).to(flat_tokens.dtype)
        print("[Step 4] Router score matrix (after sigmoid):", router_scores.shape)

        # Broadcast indices for each expert to repeat all tokens
        repeat_indices = torch.arange(flat_tokens.size(0), device=flat_tokens.device).expand(self.num_experts, -1)
        print("[Step 5] Repeat indices shape:", repeat_indices.shape)

        # Repeat token vectors for all experts
        expert_inputs = flat_tokens[repeat_indices.flatten()]  # [E * B * L, D]
        print("[Step 6] Expert inputs shape (before masking):", expert_inputs.shape)

        # Mask inputs: zero out tokens not selected for this expert
        masked_inputs = expert_inputs * router_scores.flatten().unsqueeze(-1)
        print("[Step 7] Expert inputs shape (after masking):", masked_inputs.shape)

        # Expert FFN
        expert_outputs = self.experts(masked_inputs)
        print("[Step 8] Expert outputs shape:", expert_outputs.shape)

        # Shared MLP (dense path)
        shared_output = self.shared_expert(flat_tokens)
        print("[Step 9] Shared MLP output shape:", shared_output.shape)

        # Add expert outputs into shared using scatter_add
        router_flat = repeat_indices.flatten().unsqueeze(-1).expand(-1, self.hidden_dim)
        shared_output.scatter_add_(0, router_flat, expert_outputs)
        print("[Step 10] Output after combining expert + shared:", shared_output.shape)

        return shared_output.view(batch_size, seq_len, -1)
