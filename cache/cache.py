import torch

class Cache:
    def __init__(self, num_layers):
        self._seen_tokens = 0
        self.key_cache = [torch.tensor([]) for _ in range(num_layers)]
        self.value_cache = [torch.tensor([]) for _ in range(num_layers)]

    def update(self, key_states, value_states, layer_idx):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
        self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_len(self, layer_idx=0):
        return self.key_cache[layer_idx].shape[-2] if self.key_cache[layer_idx].numel() != 0 else 0
