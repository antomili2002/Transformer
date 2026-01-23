import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import yaml
import time
from pathlib import Path
from tqdm import tqdm
import wandb

from modelling.model import Transformer
from modelling.transformer_preln import PreLNTransformer
from modelling.dataloader import TranslationDataset, MyBPETokenizer
from modelling.optimizer import create_adamw_optimizer
from modelling.scheduler import TransformerScheduler
from translate import evaluate_bleu


def load_config(config_path=None):
    file_path = 'config_preln.yaml' # config_preln or config_postln
    if config_path is None:
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        # Config is one level up from the script directory
        config_path = script_dir.parent / file_path

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

    # Prepare tokenizer
    tokenizer_dir = config['data']['tokenizer_path']
    tokenizer_file = Path(tokenizer_dir) / 'tokenizer.json'

    # Check if tokenizer already exists
    if tokenizer_file.exists():
        print(f"Loading existing tokenizer from {tokenizer_dir}")
        tokenizer_texts = ["dummy"]
    else:
        print(f"Training new tokenizer...")
        # Load tokenizer training texts
        tokenizer_text_file = Path(config['data']['train_path']).parent / 'cleaned_wmt17_de_en_texts_for_tokenizer.json'
        with open(tokenizer_text_file, 'r', encoding='utf-8') as f:
            tokenizer_texts = json.load(f)
        print(f"Training tokenizer on {len(tokenizer_texts)} sentences...")

    tokenizer = MyBPETokenizer(
        texts=tokenizer_texts,
        vocab_size=config['model']['vocab_size'],
        save_dir=tokenizer_dir
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

    model = PreLNTransformer(
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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
        learning_rate=config['training']['learning_rate'],
        betas=(config['training']['adam_beta1'], config['training']['adam_beta2']),
        eps=config['training']['adam_eps'],
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

    total_steps = config['training']['num_steps']
    log_interval = config['training'].get('log_interval', 100)
    val_interval = config['training'].get('val_interval', 5000)
    checkpoint_interval = config['training'].get('checkpoint_interval', 10000)

    print(f"Training for {total_steps:,} steps")
    print(f"Logging every {log_interval} steps, validating every {val_interval} steps")

    global_step = 0
    running_loss = 0.0
    model.train()

    train_iter = iter(train_loader)
    pbar = tqdm(total=total_steps, desc='Training')

    step_times = []
    log_timing = config.get('ablation', {}).get('log_timing', False)
    training_start_time = time.time() if log_timing else None

    while global_step < total_steps:
        step_start = time.time() if log_timing else None
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        src_ids = batch["src_ids"].to(device)
        tgt_ids = batch["tgt_ids"].to(device)

        tgt_input = tgt_ids[:, :-1]
        tgt_output = tgt_ids[:, 1:]

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

        if config['training'].get('max_grad_norm', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['max_grad_norm']
            )

        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        global_step += 1

        if log_timing:
            step_time = time.time() - step_start
            step_times.append(step_time)

        pbar.update(1)
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })

        if global_step % log_interval == 0:
            avg_loss = running_loss / log_interval

            log_dict = {
                'step': global_step,
                'train_loss': avg_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            }

            if log_timing and len(step_times) > 0:
                log_dict['time_per_step'] = np.mean(step_times)
                log_dict['samples_per_sec'] = config['training']['batch_size'] / np.mean(step_times)
                step_times = []

            if config['wandb']['enabled']:
                wandb.log(log_dict)

            running_loss = 0.0

        if global_step % val_interval == 0:
            val_loss = validate(model, val_loader, criterion, device, tokenizer)

            print(f"\nStep {global_step:,}/{total_steps:,} | Val Loss: {val_loss:.4f}")

            log_dict = {'step': global_step, 'val_loss': val_loss}

            if config['wandb']['enabled']:
                wandb.log(log_dict)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scheduler, global_step, val_loss,
                    config, f"{config['paths']['checkpoint_dir']}/best_model.pt"
                )
                print(f"Saved best model (val_loss: {val_loss:.4f})")

            model.train()

        if global_step % checkpoint_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, global_step, loss.item(),
                config, f"{config['paths']['checkpoint_dir']}/model_step_{global_step}.pt"
            )
            print(f"Saved checkpoint at step {global_step:,}")

    pbar.close()

    if log_timing:
        total_training_time = time.time() - training_start_time
        print(f"Total training time: {total_training_time/3600:.2f} hours")

    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    print("\nEvaluating on test set...")
    eval_start_time = time.time() if log_timing else None
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
        device=device,
        config=config,
        num_samples=None  # Already limited above if needed
    )

    test_bleu = results['bleu_score']
    print(f"Test BLEU Score: {test_bleu:.4f}")

    if log_timing:
        eval_time = time.time() - eval_start_time
        total_time = total_training_time + eval_time
        print(f"Evaluation time: {eval_time/60:.2f} minutes")
        print(f"Total time: {total_time/3600:.2f} hours")

    if config['wandb']['enabled']:
        log_dict = {'test_bleu_score': test_bleu}

        if log_timing:
            log_dict['total_training_time_hours'] = total_training_time / 3600
            log_dict['eval_time_minutes'] = eval_time / 60
            log_dict['total_time_hours'] = total_time / 3600

        wandb.log(log_dict)

        columns = ["src", "tgt", "pred"]
        translation_table = wandb.Table(columns=columns)

        for src, ref, pred in zip(
            results['source_texts'],
            results['references'],
            results['predictions']
        ):
            translation_table.add_data(src, ref, pred)

        wandb.log({"test_translations": translation_table})
        wandb.finish()


if __name__ == "__main__":
    main()
