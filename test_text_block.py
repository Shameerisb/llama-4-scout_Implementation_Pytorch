import torch
from ffn.text_block import Llama4TextBlock
from types import SimpleNamespace

# Dummy config matching your earlier definitions
config = SimpleNamespace(
    hidden_size=5120,
    intermediate_size=8192,         # expert feedforward size
    intermediate_size_mlp=16384,    # shared MLP size
    num_local_experts=16,
    num_experts_per_tok=1
)


# Random input simulating attention output
x = torch.randn(1, 10, config.hidden_size)

# Create and run the text block
block = Llama4TextBlock(config)
output = block(x)

print("\nFinal output shape:", output.shape)
