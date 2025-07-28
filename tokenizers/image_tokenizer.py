# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms
# from PIL import Image


# class Llama4ImageTokenizer(nn.Module):
#     def __init__(self, image_size=448, patch_size=14, input_channels=3, hidden_dim=768):
#         super().__init__()
#         assert image_size % patch_size == 0, "Image size must be divisible by patch size."
#         self.image_size = image_size
#         self.patch_size = patch_size
#         self.input_channels = input_channels
#         self.hidden_dim = hidden_dim
#         self.num_patches = (image_size // patch_size) ** 2

#         # Linear projection of flattened patches (Conv2D as patch embedding)
#         self.patch_embed = nn.Conv2d(
#             in_channels=input_channels,
#             out_channels=hidden_dim,
#             kernel_size=patch_size,
#             stride=patch_size
#         )

#         # Positional embedding (learned or rotary will be added later in model)
#         self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))

#         # Image preprocessing pipeline
#         self.preprocess = transforms.Compose([
#             transforms.Resize((image_size, image_size)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.5] * input_channels,
#                 std=[0.5] * input_channels
#             )
#         ])

#     def forward(self, image: Image.Image):
#         """
#         image: PIL.Image or batched Tensor
#         return: (B, num_patches, hidden_dim)
#         """
#         if isinstance(image, Image.Image):
#             image = self.preprocess(image).unsqueeze(0)  # (1, 3, H, W)

#         x = self.patch_embed(image)                    # (B, C, H/ps, W/ps)
#         x = x.flatten(2).transpose(1, 2)               # (B, num_patches, hidden_dim)
#         x = x + self.position_embeddings               # add positional encodings

#         return x




import torch
import torch.nn as nn
import torchvision.transforms as T

class Llama4ImageTokenizer(nn.Module):
    def __init__(self, patch_size=16, embed_dim=4096):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.linear = nn.Linear(3 * patch_size * patch_size, embed_dim)

    def forward(self, image):
        print("Received image:", type(image))
        
        # Step 1: Convert PIL to tensor
        tensor = T.ToTensor()(image)  # (3, H, W)
        print("Step 1 - After ToTensor():", tensor.shape)

        # Step 2: Add batch dimension
        tensor = tensor.unsqueeze(0)  # (1, 3, H, W)
        print("Step 2 - After unsqueeze (batch dim added):", tensor.shape)

        # Step 3: Extract image shape
        B, C, H, W = tensor.shape
        ph, pw = self.patch_size, self.patch_size

        assert H % ph == 0 and W % pw == 0, "Image size must be divisible by patch size"

        # Step 4: Unfold height and width
        tensor = tensor.unfold(2, ph, ph).unfold(3, pw, pw)  # (1, 3, H//ph, W//pw, ph, pw)
        print("Step 4 - After unfolding:", tensor.shape)

        # Step 5: Rearrange into patches
        tensor = tensor.permute(0, 2, 3, 1, 4, 5)  # (1, H//ph, W//pw, 3, ph, pw)
        print("Step 5 - After permute:", tensor.shape)

        tensor = tensor.reshape(B, -1, C * ph * pw)  # (1, num_patches, patch_dim)
        print("Step 6 - After flattening patches:", tensor.shape)

        # Step 6: Linear projection to embedding dim
        embedded = self.linear(tensor)  # (1, num_patches, embed_dim)
        print("Step 7 - After linear projection:", embedded.shape)

        return embedded
