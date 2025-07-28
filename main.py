from configs.text_config import Llama4TextConfig
from configs.vision_config import Llama4VisionConfig
from multimodal.llama4_multimodal_wrapper import Llama4ForConditionalGeneration
import torch

if __name__ == "__main__":
    text_config = Llama4TextConfig(hidden_size=768, intermediate_size=768*4)
    vision_config = Llama4VisionConfig(image_size=448, patch_size=14, num_hidden_layers=2)
    
    model = Llama4ForConditionalGeneration(vision_config, text_config)
    
    input_ids = torch.randint(0, 200000, (4, 2048))
    prepend_image_tokens = torch.tensor([200080] + [200092]*256 + [200081]).unsqueeze(0).expand(4, -1)
    input_ids = torch.cat([prepend_image_tokens, input_ids], dim=-1)

    pixel_values = torch.randn(4, 3, 448, 448)
    outputs, cache = model(input_ids=input_ids, pixel_values=pixel_values)

    print("Next token prediction:", outputs[:, -1].argmax(dim=-1))
