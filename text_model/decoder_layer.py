import torch
import torch.nn as nn
from attention.text_attention import Llama4TextAttention
from ffn.moe import Llama4TextMoe
from ffn.mlp import Llama4TextMLP
from norm.rms_norm import Llama4RMSNorm

class Llama4DecoderLayer(nn.Module):
    """
    Single transformer decoder block in the LLaMA 4 architecture.
    Includes attention, optional MoE or dense FFN, and norm layers.
    """
    def __init__(self, layer_id, config):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size

        self.attn = Llama4TextAttention(config=config, layer_id=layer_id)
        self.attn_norm = Llama4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if (layer_id % 2) == 0:
            # Use MoE in even-numbered layers
            self.ffn = Llama4TextMoe(config)
        else:
            # Use dense MLP in odd-numbered layers
            self.ffn = Llama4TextMLP(config)

        self.ffn_norm = Llama4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        """
        Forward pass through a single decoder layer.

        Args:
            hidden_states: (B, T, D)
            attention_mask: (B, T)
            position_ids: (B, T)
            past_key_value: (optional tuple of cached K/V)

        Returns:
            hidden_states: Updated hidden states
            past_kv: Updated past K/V for caching
        """
        # === Attention Block ===
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, past_kv = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value
        )
        hidden_states = hidden_states + residual  # Residual connection

        # === Feedforward Block ===
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = hidden_states + residual  # Residual connection

        return hidden_states, past_kv
