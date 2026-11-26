
from pathlib import Path

from data.dataset import MathTextDataset
from data.tokenizer import MathTokenizer


def inspect():
    tokenizer = MathTokenizer.load("models/tokenizer.json")

    # Re-encode the text using the saved tokenizer.
    text = Path("data/textbook.txt").read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    token_ids = [
        tokenizer.token_id(tokenizer.BOS),
        *token_ids,
        tokenizer.token_id(tokenizer.EOS),
    ]

    dataset = MathTextDataset(
        token_ids=token_ids,
        seq_len=128,
        pad_token_id=tokenizer.token_id(tokenizer.PAD),
    )

    sample = dataset[0]

    print(f"Dataset samples: {len(dataset):,}")
    print(f"Input shape: {tuple(sample['input_ids'].shape)}")
    print(f"Label shape: {tuple(sample['labels'].shape)}")

    print("\nDecoded input:")
    print(tokenizer.decode(sample["input_ids"].tolist()))

    print("\nDecoded target:")
    print(tokenizer.decode(sample["labels"].tolist()))


if __name__ == "__main__":
    inspect()
