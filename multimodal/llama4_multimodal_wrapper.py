import torch
import torch.nn as nn
from vision_model.vision_model import Llama4VisionModel
from vision_model.projector import Llama4MultiModalProjector
from .llama4_for_causal_lm import Llama4ForCausalLM

class Llama4ForConditionalGeneration(nn.Module):
    """
    Combines vision and language backbones for multimodal text generation.
    Injects projected vision embeddings into the text sequence.
    """
    def __init__(self, 
                 vision_config, 
                 text_config,
                 boi_token_index=200080, 
                 eoi_token_index=200081, 
                 image_token_index=200092):
        super().__init__()
        self.vision_model = Llama4VisionModel(vision_config)
        self.projector = Llama4MultiModalProjector(vision_config, text_config)
        self.language_model = Llama4ForCausalLM(text_config)

        self.vocab_size = text_config.vocab_size
        self.pad_token_id = text_config.pad_token_id
        self.boi_token_index = boi_token_index
        self.eoi_token_index = eoi_token_index
        self.image_token_index = image_token_index

    def forward(self,
                input_ids, 
                pixel_values=None,
                attention_mask=None, 
                position_ids=None, 
                past_key_values=None):
        
        token_embeddings = self.language_model.model.embed_tokens(input_ids)

        if pixel_values is not None:
            image_features = self.vision_model(pixel_values)       # (B, N, D)
            image_flat = image_features.view(-1, image_features.shape[-1])
            projected_image = self.projector(image_flat)

            # Find placeholder tokens in input
            mask = (input_ids == self.image_token_index).unsqueeze(-1)  # (B, T, 1)
            B, T, D = token_embeddings.shape

            # Flatten everything for `masked_scatter_`
            token_embeddings = token_embeddings.view(-1, D)
            flat_mask = mask.view(-1).bool()
            assert flat_mask.sum() == projected_image.shape[0]

            expanded_mask = flat_mask.unsqueeze(-1).expand(-1, D)
            token_embeddings.masked_scatter_(expanded_mask, projected_image)
            token_embeddings = token_embeddings.view(B, T, D)

        logits, past = self.language_model(
            input_embeds=token_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values
        )

        return logits, past
