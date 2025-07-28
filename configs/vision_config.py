from dataclasses import dataclass

@dataclass
class Llama4VisionConfig:
    image_size: int = 448
    patch_size: int = 14
    num_channels: int = 3
    hidden_size: int = 768
    intermediate_size: int = 768 * 4
    num_attention_heads: int = 12
    num_hidden_layers: int = 2
    attention_dropout: float = 0.1
    projector_output_dim: int = 768
    pixel_shuffle_ratio: int = 2
    rope_theta: float = 10000.0  # typical RoPE frequency scale for vision
