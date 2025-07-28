import torch.nn as nn

class Llama4TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, pad_token_id):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)
