import torch.nn as nn

class Llama4UnfoldConvolution(nn.Module):
    """
    Converts an image into non-overlapping patches using `nn.Unfold`
    and projects each patch into the model embedding space.
    """
    def __init__(self, config):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=config.patch_size, stride=config.patch_size)
        self.linear = nn.Linear(config.num_channels * config.patch_size ** 2, config.hidden_size, bias=False)

    def forward(self, x):
        # x: (B, C, H, W)
        patches = self.unfold(x)                   # (B, C*patch^2, Num_Patches)
        patches = patches.transpose(1, 2)          # (B, Num_Patches, Patch_Embed)
        return self.linear(patches)                # (B, Num_Patches, Hidden_Dim)
