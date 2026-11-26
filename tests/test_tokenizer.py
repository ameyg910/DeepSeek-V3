from data.tokenizer import MathTokenizer

def test_tokenizer_training():
    texts = [
        "The derivative of x^2 is 2x.",
        "The derivative of x^3 is 3x^2",
    ] * 10

    tokenizer = MathTokenizer()
    tokenizer.train(texts, vocab_size=300)

    assert tokenizer.vocab_size > 0

def test_special_tokens():
    texts = ["hello world"] * 10 

    tokenizer = MathTokenizer()
    tokenizer.train(texts, vocab_size=300)

    token_ids = tokenizer.encode("hello world")

    assert token_ids[0] == tokenizer.token_id("[BOS]")
    assert token_ids[-1] == tokenizer.token_id("[EOS]")

def test_encode_decode():
    texts = [
        "The derivative of x^2 is 2x.",
        "Integration is the inverse of differentiation.",
    ] * 10

    tokenizer = MathTokenizer()
    tokenizer.train(texts, vocab_size=300)

    text = "The derivative of x^2 is 2x."
    token_ids = tokenizer.encode(text, add_special_tokens=False,)
    decoded = tokenizer.decode(token_ids)
    assert decoded == text

def test_save_and_load(tmp_path):
    texts = ["hello world"] * 10

    tokenizer = MathTokenizer()
    tokenizer.train(texts, vocab_size=300)

    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = MathTokenizer.load(path)
    text = "hello world"

    original_ids = tokenizer.encode(text)
    loaded_ids = loaded.encode(text)

    assert original_ids == loaded_ids