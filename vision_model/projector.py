import torch.nn as nn

class Llama4MultiModalProjector(nn.Module):
    """
    Projects vision embeddings into the same space as the text transformer.
    """
    def __init__(self, vision_config, text_config):
        super().__init__()
        self.linear = nn.Linear(vision_config.projector_output_dim, text_config.hidden_size, bias=False)

    def forward(self, x):
        return self.linear(x)
