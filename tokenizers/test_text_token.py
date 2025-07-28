# from text_tokenizer import Tokenizer
# from pathlib import Path  

# def test_tokenizer():
#     model_path = Path(r"C:\TALHA\OneDrive - National University of Sciences & Technology\Desktop\llm\llama4\tokenizers\tokenizer.model") 
#     tokenizer = Tokenizer(model_path)

#     input_text = "LLaMA 4 is a powerful model."
#     encoded = tokenizer.encode(input_text, bos=True, eos=True)
#     decoded = tokenizer.decode(encoded)

#     print(f"Input Text: {input_text}")
#     print(f"Encoded Token IDs: {encoded}")
#     print(f"Decoded Text: {decoded}")
#     print(f"BOS ID: {tokenizer.bos_id}")
#     print(f"EOS ID: {tokenizer.eos_id}")
#     print(f"PAD ID: {tokenizer.pad_id}")
#     print(f"Vocab Size: {tokenizer.n_words}")

# if __name__ == "__main__":
#     test_tokenizer()




from text_tokenizer import Llama4TextTokenizer
from pathlib import Path

def test_tokenizer():
    model_path = Path(r"C:\TALHA\OneDrive - National University of Sciences & Technology\Desktop\llm\llama4\tokenizers\tokenizer1.model")
    tokenizer = Llama4TextTokenizer(model_path)

    text = "LLaMA 4 is a powerful model."
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    tokens = tokenizer.get_tokens(text)

    print(f"Input Text: {text}")
    print(f"Encoded Token IDs: {encoded}")
    print(f"Subword Tokens: {tokens}")
    print(f"Decoded Text: {decoded}")
    print(f"BOS ID: {tokenizer.bos_id}")
    print(f"EOS ID: {tokenizer.eos_id}")
    print(f"PAD ID: {tokenizer.pad_id}")
    print(f"UNK ID: {tokenizer.unk_id}")
    print(f"Vocab Size: {tokenizer.vocab_size}")

if __name__ == "__main__":
    test_tokenizer()
