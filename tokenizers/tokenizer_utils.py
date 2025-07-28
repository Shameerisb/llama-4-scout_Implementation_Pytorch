import base64

def load_bpe_file(model_path):
    """
    Loads a TikToken-style mergeable_ranks BPE file from .model (text file).
    Returns:
        dict: {bytes: int}
    """
    with open(model_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line and not line.startswith("#")]
    return {base64.b64decode(line): i for i, line in enumerate(lines)}
