import pytest 
import torch 

from data.dataset import MathTextDataset

def test_dataset_shifts_tokens():
    dataset = MathTextDataset(
        token_ids=[10, 11, 12, 13, 14],
        seq_len=4,
        pad_token_id=0,
    )

    sample = dataset[0]

    assert sample["input_ids"].tolist() == [10, 11, 12, 13]
    assert sample["labels"].tolist() == [11, 12, 13, 14]

def test_dataset_length():
    dataset = MathTextDataset(
        token_ids=list(range(10)),
        seq_len=4,
        pad_token_id=0,
    )

    # Chunks of 5 tokens: [0..4] and [5..9].
    assert len(dataset) == 2

def test_last_chunk_is_padded_and_ignored():
    dataset = MathTextDataset(
        token_ids=[10, 11, 12, 13, 14, 15],
        seq_len=4,
        pad_token_id=0,
    )

    sample = dataset[1]

    assert sample["input_ids"].tolist() == [15, 0, 0, 0]
    assert sample["labels"].tolist() == [-100, -100, -100, -100]

def test_dataset_returns_long_tensors():
    dataset = MathTextDataset(
        token_ids=[10, 11, 12],
        seq_len=2,
        pad_token_id=0,
    )

    sample = dataset[0]

    assert sample["input_ids"].dtype == torch.long
    assert sample["labels"].dtype == torch.long

def test_invalid_inputs():
    with pytest.raises(ValueError):
        MathTextDataset([], seq_len=4, pad_token_id=0)

    with pytest.raises(ValueError):
        MathTextDataset([1, 2], seq_len=0, pad_token_id=0)