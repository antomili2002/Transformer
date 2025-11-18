import torch
import json
from torch import nn
from torch.utils.data import DataLoader, Dataset
from modelling.dataloader import TranslationDataset, MyBPETokenizer

def create_dataloaders(tokenizer, batch_size=32, max_src_len=64, max_tgt_len=64):
    # Load cleaned datasets
    with open("data/cleaned_wmt17_de_en_split_train_10000.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open("data/cleaned_wmt17_de_en_split_test_2000.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    with open("data/cleaned_wmt17_de_en_split_validation_2000.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)

    train_dataset = TranslationDataset(train_data, tokenizer, max_src_len, max_tgt_len)
    test_dataset = TranslationDataset(test_data, tokenizer, max_src_len, max_tgt_len)
    val_dataset = TranslationDataset(val_data, tokenizer, max_src_len, max_tgt_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader

def main():
    # Instantiate tokenizer
    with open("data/cleaned_wmt17_de_en_texts_for_tokenizer_120000.json", "r", encoding="utf-8") as f:
        train_texts = json.load(f)
    tokenizer = MyBPETokenizer(texts=train_texts, vocab_size=50000, save_dir="./my_bpe_tokenizer")

    # Create dataloaders
    train_loader, test_loader, val_loader = create_dataloaders(tokenizer, batch_size=32)

    print("Number of training batches:", len(train_loader))
    print("Number of testing batches:", len(test_loader))
    print("Number of validation batches:", len(val_loader))