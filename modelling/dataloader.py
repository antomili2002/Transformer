import torch
import os
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
from transformers import GPT2Tokenizer

class MyBPETokenizer():
    def __init__(self, texts, vocab_size=50000, save_dir="./my_bpe_tokenizer"):
        """Tokenizer using BPE from the tokenizers library.

        Args:
            texts list: lists of strings cleaned for training the tokenizer
            vocab_size (int, optional): Vocabulary size. Defaults to 50000.
            save_dir (str, optional): Directory to save the tokenizer files. Defaults to "./my_bpe_tokenizer".
        """
        self.vocab_size = vocab_size
        self.save_dir = save_dir
        
        self.pad = "[PAD]"
        self.bos = "[BOS]"
        self.eos = "[EOS]"
        self.unk = "[UNK]"
        
        self.special_tokens = [self.pad, self.bos, self.eos, self.unk]
        
        # train HF tokenizer
        self._train_bpe(texts)
        
        # load GPT2 tokenizer
        self._load_gpt2_tokenizer()
        
    def _train_bpe(self, texts):
        tokenizer = Tokenizer(BPE(unk_token=self.unk))
        tokenizer.normalizer = Sequence([NFD(), Lowercase()])
        tokenizer.pre_tokenizer = Whitespace()
        
        trainer = BpeTrainer(vocab_size=self.vocab_size, 
                             special_tokens=self.special_tokens)
        
        tokenizer.train_from_iterator(texts, trainer=trainer)
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        tokenizer.model.save((self.save_dir))
    
    def _load_gpt2_tokenizer(self):
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(self.save_dir)
        self.gpt2_tokenizer.pad_token = self.pad
        self.gpt2_tokenizer.bos_token = self.bos
        self.gpt2_tokenizer.eos_token = self.eos
        self.gpt2_tokenizer.unk_token = self.unk
        
    @property
    def pad_id(self):
        return self.gpt2_tokenizer.pad_token_id

    @property
    def bos_id(self):
        return self.gpt2_tokenizer.bos_token_id

    @property
    def eos_id(self):
        return self.gpt2_tokenizer.eos_token_id

    @property
    def unk_id(self):
        return self.gpt2_tokenizer.unk_token_id

    # Encoding for source / target sequences
    def encode_src(self, text, max_length=64):
        return self._encode_with_bos_eos(text, max_length)

    def encode_tgt(self, text, max_length=64):
        return self._encode_with_bos_eos(text, max_length)

    def _encode_with_bos_eos(self, text, max_length):
        # Get plain token IDs (no special tokens from GPT2)
        ids = self.gpt2_tokenizer.encode(text, add_special_tokens=False)

        # Add BOS and EOS
        ids = [self.bos_id] + ids + [self.eos_id]

        # Truncate if too long
        if len(ids) > max_length:
            ids = ids[:max_length]

        # Pad if too short
        pad_len = max_length - len(ids)
        if pad_len > 0:
            ids = ids + [self.pad_id] * pad_len

        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        # Decode while skipping special tokens
        return self.gpt2_tokenizer.decode(ids, skip_special_tokens=True)

class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_src_len=64, max_tgt_len=64):
        """
        data: list of dicts with keys "src", "tgt"
        tokenizer: instance of MyBPETokenizer
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        src_text = sample["src"]
        tgt_text = sample["tgt"]

        src_ids = self.tokenizer.encode_src(src_text, max_length=self.max_src_len)
        tgt_ids = self.tokenizer.encode_tgt(tgt_text, max_length=self.max_tgt_len)

        return {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
        }