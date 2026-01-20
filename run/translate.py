import torch
import torch.nn as nn
import yaml
import json
from pathlib import Path
from typing import Optional, List
from evaluate import load

from modelling.model import Transformer
from modelling.dataloader import MyBPETokenizer


def greedy_decode(
    model: nn.Module,
    src: torch.Tensor,
    src_mask: Optional[torch.Tensor],
    max_len: int,
    start_token_id: int,
    end_token_id: int,
    device: torch.device
):
    model.eval()

    # encode source sentence once
    with torch.no_grad():
        src = src.to(device)
        if src_mask is not None:
            src_mask = src_mask.to(device)

        # Encode: [1, src_len] -> [1, src_len, d_model]
        memory = model.encode(src, src_mask)

    # initialize decoder input with start token [BOS]
    tgt = torch.tensor([[start_token_id]], dtype=torch.long, device=device)

    # Generate tokens autoregressively
    for i in range(max_len - 1): 
        with torch.no_grad():
            output = model.decode(tgt, memory, tgt_mask=None, memory_mask=src_mask)
            logits = model.output_projection(output)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
            if next_token.item() == end_token_id:
                break
    return tgt


def load_config(config_path=None):
    if config_path is None:
        script_dir = Path(__file__).parent
        config_path = script_dir.parent / 'config.yaml'

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(checkpoint_path, config, device):
    tokenizer = MyBPETokenizer(
        texts=["dummy"],
        vocab_size=config['model']['vocab_size'],
        save_dir=config['data']['tokenizer_path']
    )

    model = Transformer(
        vocab_size=len(tokenizer.tokenizer.get_vocab()),
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        max_len=config['model']['max_len'],
        dropout=config['model']['dropout']
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}, Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")

    return model, tokenizer


def translate_sentence(model, tokenizer, source_text, device, config):
    model.eval()

    with torch.no_grad():
        src_ids = tokenizer.encode_src(
            source_text,
            max_length=config['data']['max_src_len']
        ).unsqueeze(0).to(device)

        src_mask = (src_ids != tokenizer.pad_id).to(device)

        output_ids = greedy_decode(
            model=model,
            src=src_ids,
            src_mask=src_mask,
            max_len=config['generation']['max_length'],
            start_token_id=tokenizer.bos_id,
            end_token_id=tokenizer.eos_id,
            device=device
        )

        output_tokens = output_ids[0].cpu().tolist()
        translation = tokenizer.decode(output_tokens)

    return translation


def evaluate_bleu(model, test_data, tokenizer, device, config, num_samples=None):
    bleu = load("bleu")
    model.eval()

    if num_samples is not None:
        test_data = test_data[:num_samples]

    predictions = []
    references = []

    print(f"Generating translations for {len(test_data)} samples...")

    for i, sample in enumerate(test_data):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(test_data)}")

        source_text = sample['src']
        reference_text = sample['tgt']

        translation = translate_sentence(model, tokenizer, source_text, device, config)

        predictions.append(translation)
        references.append([reference_text])

    bleu_results = bleu.compute(predictions=predictions, references=references)

    return {
        'bleu_score': bleu_results['bleu'],
        'predictions': predictions,
        'references': [ref[0] for ref in references],
        'source_texts': [sample['src'] for sample in test_data]
    }


def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    checkpoint_path = Path(config['paths']['checkpoint_dir']) / 'best_model.pt'

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    model, tokenizer = load_model_and_tokenizer(checkpoint_path, config, device)

    with open(config['data']['test_path'], 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    num_samples = config['data'].get('num_test_samples', 10)
    if num_samples is None:
        num_samples = 10

    test_data = test_data[:num_samples]

    print(f"Translating {len(test_data)} test samples...\n")

    for i, sample in enumerate(test_data, 1):
        source = sample['src']
        target = sample['tgt']
        translation = translate_sentence(model, tokenizer, source, device, config)

        print(f"[{i}/{len(test_data)}]")
        print(f"Source: {source}")
        print(f"Target: {target}")
        print(f"Translation: {translation}")


if __name__ == "__main__":
    main()
