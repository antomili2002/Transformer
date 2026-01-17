import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import yaml
from pathlib import Path
from tqdm import tqdm
import wandb

from modelling.model import Transformer
from modelling.dataloader import TranslationDataset, MyBPETokenizer
from modelling.optimizer import create_adamw_optimizer
from modelling.scheduler import TransformerScheduler
from modelling.generation import evaluate_bleu


def load_config(config_path='../config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(config):
    with open(config['data']['train_path'], 'r', encoding='utf-8') as f:
        train_data = json.load(f)
        if config['data']['num_train_samples'] is not None:
            train_data = train_data[:config['data']['num_train_samples']]

    with open(config['data']['val_path'], 'r', encoding='utf-8') as f:
        val_data = json.load(f)
        if config['data']['num_val_samples'] is not None:
            val_data = val_data[:config['data']['num_val_samples']]

    with open(config['data']['tokenizer_path'] + '/vocab.json', 'r') as f:
        tokenizer_texts = ["dummy"]

    tokenizer = MyBPETokenizer(
        texts=tokenizer_texts,
        vocab_size=config['model']['vocab_size'],
        save_dir=config['data']['tokenizer_path']
    )

    train_dataset = TranslationDataset(
        train_data, tokenizer,
        config['data']['max_src_len'],
        config['data']['max_tgt_len']
    )

    val_dataset = TranslationDataset(
        val_data, tokenizer,
        config['data']['max_src_len'],
        config['data']['max_tgt_len']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    return train_loader, val_loader, tokenizer


def create_model(config, tokenizer, device):
    vocab_size = tokenizer.tokenizer.get_vocab_size()

    model = Transformer(
        vocab_size=vocab_size,
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        max_len=config['model']['max_len'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout']
    ).to(device)

    return model


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, config, tokenizer):
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        src_ids = batch["src_ids"].to(device)
        tgt_ids = batch["tgt_ids"].to(device)

        tgt_input = tgt_ids[:, :-1]
        tgt_output = tgt_ids[:, 1:]

        # Create padding masks
        src_mask = (src_ids != tokenizer.pad_id)
        tgt_mask = (tgt_input != tokenizer.pad_id)
        memory_mask = src_mask

        optimizer.zero_grad()
        output = model(src_ids, tgt_input,
                      src_mask=src_mask,
                      tgt_mask=tgt_mask,
                      memory_mask=memory_mask)

        vocab_size = output.shape[-1]
        loss = criterion(
            output.reshape(-1, vocab_size),
            tgt_output.reshape(-1)
        )

        loss.backward()

        if config['training']['max_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['max_grad_norm']
            )

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, tokenizer):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            src_ids = batch["src_ids"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)

            tgt_input = tgt_ids[:, :-1]
            tgt_output = tgt_ids[:, 1:]

            # Create padding masks (same as training)
            src_mask = (src_ids != tokenizer.pad_id)
            tgt_mask = (tgt_input != tokenizer.pad_id)
            memory_mask = src_mask

            output = model(src_ids, tgt_input,
                          src_mask=src_mask,
                          tgt_mask=tgt_mask,
                          memory_mask=memory_mask)

            vocab_size = output.shape[-1]
            loss = criterion(
                output.reshape(-1, vocab_size),
                tgt_output.reshape(-1)
            )

            total_loss += loss.item()

    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, config, path):
    Path(config['paths']['checkpoint_dir']).mkdir(exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'config': config
    }

    torch.save(checkpoint, path)


def main():
    config = load_config()
    set_seed(config['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if config['wandb']['enabled']:
        wandb.init(
            project=config['wandb']['project'],
            config=config
        )

    print("Loading data...")
    train_loader, val_loader, tokenizer = load_data(config)

    print("Creating model...")
    model = create_model(config, tokenizer, device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = create_adamw_optimizer(
        model,
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = TransformerScheduler(
        optimizer,
        d_model=config['model']['d_model'],
        warmup_steps=config['training']['warmup_steps']
    )

    if config['wandb']['enabled']:
        wandb.watch(model, log='all', log_freq=100)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print(f"Training for {config['training']['num_epochs']} epochs...")

    for epoch in range(config['training']['num_epochs']):
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")

        train_loss = train_epoch(
            model, train_loader, criterion,
            optimizer, scheduler, device, config, tokenizer
        )

        val_loss = validate(model, val_loader, criterion, device, tokenizer)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if config['wandb']['enabled']:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                config, f"{config['paths']['checkpoint_dir']}/best_model.pt"
            )
            print(f"Saved best model (val_loss: {val_loss:.4f})")

        save_checkpoint(
            model, optimizer, scheduler, epoch, val_loss,
            config, f"{config['paths']['checkpoint_dir']}/model_epoch_{epoch+1}.pt"
        )

    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    print("\nEvaluating on test set...")
    with open(config['data']['test_path'], 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        if config['data']['num_test_samples'] is not None:
            test_data = test_data[:config['data']['num_test_samples']]

    checkpoint = torch.load(config['paths']['checkpoint_dir'] + '/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    results = evaluate_bleu(
        model=model,
        test_data=test_data,
        tokenizer=tokenizer,
        max_len=config['generation']['max_length'],
        device=device,
        num_samples=None  # Already limited above if needed
    )

    test_bleu = results['bleu_score']
    print(f"Test BLEU Score: {test_bleu:.2f}")

    if config['wandb']['enabled']:
        wandb.log({'test_bleu_score': test_bleu})
        wandb.finish()


if __name__ == "__main__":
    main()
