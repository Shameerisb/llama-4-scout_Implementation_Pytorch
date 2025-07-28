import torch
import torch.nn as nn
from text_model.text_model import Llama4TextModel

class Llama4ForCausalLM(nn.Module):
    """
    Causal Language Modeling head over the LLaMA 4 text backbone.
    Outputs logits for next-token prediction.
    """
    def __init__(self, config):
        super().__init__()
        self.model = Llama4TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, 
                input_ids=None, 
                input_embeds=None, 
                attention_mask=None, 
                position_ids=None, 
                past_key_values=None):
        
        hidden_states, past = self.model(
            input_ids=input_ids,
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        logits = self.lm_head(hidden_states)  # (B, T, Vocab)
        return logits, past
