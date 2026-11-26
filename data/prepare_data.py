
from pathlib import Path

from data.tokenizer import MathTokenizer


def main():
    textbook_path = Path("data/textbook.txt")

    if not textbook_path.exists():
        raise FileNotFoundError(
            "Run PDF extraction first to create data/textbook.txt"
        )

    text = textbook_path.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError("Extracted textbook text is empty")

    # Train the tokenizer on the textbook.
    tokenizer = MathTokenizer()
    tokenizer.train(
        texts=[text],
        vocab_size=12000,
        min_frequency=2,
    )

    tokenizer_path = Path("models/tokenizer.json")
    tokenizer.save(str(tokenizer_path))

    # Encode the full textbook without adding BOS/EOS
    # around the entire document automatically.
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    # Add document boundary tokens once.
    token_ids = [
        tokenizer.token_id(tokenizer.BOS),
        *token_ids,
        tokenizer.token_id(tokenizer.EOS),
    ]

    print(f"Text characters: {len(text):,}")
    print(f"Vocabulary size: {tokenizer.vocab_size:,}")
    print(f"Token count: {len(token_ids):,}")
    print(f"Tokenizer saved: {tokenizer_path}")

    return tokenizer, token_ids


if __name__ == "__main__":
    main()
