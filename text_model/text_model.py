import torch
import torch.nn as nn
from embeddings.embedding_layer import Llama4TokenEmbedding
from text_model.decoder_layer import Llama4TextDecoder
from norm.rms_norm import Llama4RMSNorm


class Llama4TextModel(nn.Module):
    """
    LLaMA 4 Text Transformer backbone.
    Embeds token IDs, applies rotary attention blocks, and returns hidden states.
    """
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = Llama4TokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            pad_token_id=config.pad_token_id
        )
        
        self.decoder = Llama4TextDecoder(config)
        self.norm = Llama4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self,
                input_ids=None,
                input_embeds=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None):
        """
        Args:
            input_ids: [B, T] token indices
            input_embeds: [B, T, D] optional precomputed embeddings
            attention_mask: [B, T] optional mask
            position_ids: [B, T] optional rotary position indices
            past_key_values: List of KV caches (optional for inference)
        
        Returns:
            final_hidden_states: [B, T, D]
            updated_past_kv: KV cache (if used)
        """
        if input_embeds is not None:
            hidden_states = input_embeds
        elif input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            raise ValueError("Must provide input_ids or input_embeds")

        hidden_states, past_kv = self.decoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values
        )

        hidden_states = self.norm(hidden_states)
        return hidden_states, past_kv
