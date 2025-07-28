from dataclasses import dataclass

@dataclass
class Llama4TextConfig:
    vocab_size: int = 202048
    hidden_size: int = 5120
    intermediate_size: int = 8192
    intermediate_size_mlp: int = 16384
    num_hidden_layers: int = 48
    num_attention_heads: int = 40
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 4096 * 32
    rms_norm_eps: float = 1e-5
    pad_token_id: int = 200018
    bos_token_id: int = 1
    eos_token_id: int = 2
    rope_theta: float = 500000
    attention_dropout: float = 0.0
    num_experts_per_tok: int = 1
    num_local_experts: int = 16
    use_qk_norm: bool = True
    no_rope_layer_interval: int = 4
    attention_chunk_size: int = 8192
    attn_temperature_tuning: float = 4
    floor_scale: int = 8192
    attn_scale: float = 0.1
