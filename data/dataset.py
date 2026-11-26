import torch 
from torch.utils.data import Dataset

class MathTextDataset(Dataset):
    def __init__(self, token_ids: list[int], seq_len: int, pad_token_id: int,):
        if seq_len <= 0: 
            raise ValueError("seq len cannot be negative")

        if not token_ids:
            raise ValueError("token ids cannot be empty")
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id

        chunk_size = seq_len + 1
        self.samples = []

        for start in range(0, len(token_ids), chunk_size):
            chunk = token_ids[start : start + chunk_size]

            valid_length = len(chunk)
            if valid_length < chunk_size: 
               chunk = chunk + [pad_token_id]*(chunk_size - valid_length)

            #shift by one token 
            input_ids = chunk[:-1]
            labels = chunk[1:]

            #ignore targets that correspond to padding
 
            labels = [
                token if i < valid_length - 1 else -100
                for i, token in enumerate(labels)
            ]

            self.samples.append(
                {
                    "input_ids": torch.tensor(
                        input_ids, dtype=torch.long
                    ),
                    "labels": torch.tensor(
                        labels, dtype=torch.long
                    ),
                }
            )
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]