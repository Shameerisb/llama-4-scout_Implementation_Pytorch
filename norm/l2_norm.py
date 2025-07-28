import torch
import torch.nn as nn
import torch.nn.functional as F

class Llama4TextL2Norm(nn.Module):
    """
    Applies L2 normalization over the last dimension of the text embedding.
    Typically used to normalize token embeddings or projected vectors.
    """
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # L2 normalization along last dimension
        return F.normalize(x, p=2, dim=-1, eps=self.eps)
