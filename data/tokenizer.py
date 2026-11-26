from pathlib import Path 

from tokenizers import Tokenizer 
from tokenizers.models import BPE 
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

class MathTokenizer: 
    PAD = "[PAD]"
    UNK = "[UNK]"
    BOS = "[BOS]"
    EOS = "[EOS]"

    SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]

    def __init__(self):
       self.tokenizer = None 

    def train(self, texts: list[str], vocab_size: int = 300, min_frequency: int = 2,):
        if not texts: 
            raise ValueError("training texts cannot be empty")

        if vocab_size < 260: #256 possible raw byte values (0-255) + 4 (special tokens)
            raise ValueError("vocab size must be at least 260")

        self.tokenizer = Tokenizer(
            BPE(unk_token=self.UNK)
        )
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, trim_offsets=False,)
 
        self.tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size, 
            min_frequency=min_frequency, 
            special_tokens=self.SPECIAL_TOKENS,
            initial_alphabet=ByteLevel.alphabet(), 
        )
        self.tokenizer.train_from_iterator(texts, trainer=trainer)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if self.tokenizer is None: 
            raise RuntimeError("Tokenizer has not been trained")

        encoding = self.tokenizer.encode(text)

        token_ids = encoding.ids 
        if add_special_tokens: 
            token_ids = [self.token_id(self.BOS), *token_ids, self.token_id(self.EOS),]

        return token_ids

    def decode(self, token_ids : list[int]) -> str: 
        if self.tokenizer is None: 
            raise RuntimeError("tokenizer has not been trained")
        return self.tokenizer.decode(
            token_ids, skip_special_tokens=True, 
        )
    def token_id(self, token: str) -> int:
        if self.tokenizer is None: 
            raise RuntimeError("Tokenizer has not been trained")

        token_id = self.tokenizer.token_to_id(token)

        if token_id is None:
            raise ValueError(f"Unknown special token: {token}")
        return token_id

    @property
    def vocab_size(self) -> int: 
        if self.tokenizer is None: 
            raise RuntimeError("Tokenizer has not been trained")

        return self.tokenizer.get_vocab_size()

    def save(self, path: str):
        if self.tokenizer is None: 
            raise RuntimeError("Tokenizer has not been trained")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.tokenizer.save(str(path))

    @classmethod 
    def load(cls, path: str):
        instance = cls()
        instance.tokenizer = Tokenizer.from_file(str(path))

        return instance