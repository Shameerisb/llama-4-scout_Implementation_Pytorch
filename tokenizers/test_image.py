from PIL import Image
from image_tokenizer import Llama4ImageTokenizer
import torchvision.transforms as T

# Load image and tokenizer
tokenizer = Llama4ImageTokenizer()
image = Image.open(r"C:\Users\talha\Downloads\585520-Chelsea-FC-Soccer-Field-sport-sports-soccer-stadium.jpg").convert("RGB")

# Print original size
print("PIL Image size (width, height):", image.size)

# Resize to nearest multiples of 16
new_width = image.size[0] - (image.size[0] % 16)
new_height = image.size[1] - (image.size[1] % 16)
image = image.resize((new_width, new_height))
print("Resized image to:", image.size)

# Optional: check tensor shape
image_tensor = T.ToTensor()(image)
print("Tensor shape (C, H, W):", image_tensor.shape)

# Tokenize and print token shape
tokens = tokenizer(image)
print("Tokenized image shape:", tokens.shape)  # (1, num_tokens, hidden_dim)
