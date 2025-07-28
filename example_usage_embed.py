import torch
from tokenizers.text_tokenizer import Llama4TextTokenizer
from embeddings.embedding_layer import Llama4TokenEmbedding
from embeddings.rotary_text import Llama4TextRotaryEmbedding  # <-- RoPE module

# Step 1: Dummy config
class DummyConfig:
    vocab_size = 202048
    hidden_size = 5120
    head_dim = 128  # size per head
    num_heads = 40
    rope_theta = 10000.0
    max_position_embeddings = 2048
    pad_token_id = 0

# Step 2: Load tokenizer and encode
print("Loading tokenizer...")
tokenizer = Llama4TextTokenizer(
    r"C:\TALHA\OneDrive - National University of Sciences & Technology\Desktop\llm\llama4\tokenizers\tokenizer1.model"
)

text = "This is a test sentence."
input_ids = tokenizer.encode(text)
input_tensor = torch.tensor([input_ids])  # shape: [1, L]
print("\nText:", text)
print("Token IDs:", input_ids)
print("Input Tensor Shape:", input_tensor.shape)

# Step 3: Token Embedding
embedding = Llama4TokenEmbedding(
    vocab_size=tokenizer.vocab_size,
    hidden_size=DummyConfig.hidden_size,
    pad_token_id=tokenizer.pad_id
)

output = embedding(input_tensor)  # [1, L, D]
print("\nToken Embedding Shape:", output.shape)

# Step 4: Reshape for attention heads
B, L, D = output.shape
H = DummyConfig.num_heads
D_head = DummyConfig.head_dim
assert D == H * D_head, "hidden_size must equal num_heads * head_dim"

x = output.view(B, L, H, D_head)  # [1, L, H, D_head]
print("Reshaped for attention:", x.shape)

# Step 5: Rotary Positional Embedding
rope = Llama4TextRotaryEmbedding(DummyConfig)
position_ids = torch.arange(L).unsqueeze(0)  # shape: [1, L]
print("Position IDs:", position_ids)

rotary_emb = rope(x, position_ids)  # shape: [1, L, 1, D_head//2], complex
print("Rotary Embedding shape:", rotary_emb.shape)
print("First token (head 0) - real part:", rotary_emb[0, 0, 0].real[:5])
print("First token (head 0) - imag part:", rotary_emb[0, 0, 0].imag[:5])
