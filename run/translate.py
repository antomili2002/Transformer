import torch
import torch.nn as nn
import yaml
import json
from pathlib import Path
from typing import Optional, List
from evaluate import load

from modelling.model import Transformer
from modelling.dataloader import MyBPETokenizer
from modelling.transformer_preln import PreLNTransformer


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


def beam_search_decode(
    model: nn.Module,
    src: torch.Tensor,
    src_mask: Optional[torch.Tensor],
    max_len: int,
    start_token_id: int,
    end_token_id: int,
    device: torch.device,
    beam_size: int = 5,
    length_penalty: float = 0.6
):
    model.eval()

    with torch.no_grad():
        src = src.to(device)
        if src_mask is not None:
            src_mask = src_mask.to(device)

        memory = model.encode(src, src_mask)

        # Initialize beam: (sequence, score)
        beams = [(torch.tensor([[start_token_id]], dtype=torch.long, device=device), 0.0)]
        completed = []

        for step in range(max_len - 1):
            candidates = []

            for seq, score in beams:
                # Skip if already ended
                if seq[0, -1].item() == end_token_id:
                    completed.append((seq, score))
                    continue

                # Decode
                output = model.decode(seq, memory, tgt_mask=None, memory_mask=src_mask)
                logits = model.output_projection(output)
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)

                # Get top beam_size tokens
                topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)

                # Expand beam
                for log_prob, token_id in zip(topk_log_probs[0], topk_ids[0]):
                    new_seq = torch.cat([seq, token_id.unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = score + log_prob.item()
                    candidates.append((new_seq, new_score))

            if not candidates:
                break

            # Sort by length-normalized score
            candidates = sorted(
                candidates,
                key=lambda x: x[1] / (x[0].size(1) ** length_penalty),
                reverse=True
            )

            # Keep top beam_size
            beams = candidates[:beam_size]

            # Stop if all beams ended
            if all(seq[0, -1].item() == end_token_id for seq, _ in beams):
                completed.extend(beams)
                break

        # Add remaining beams to completed
        completed.extend(beams)

        if not completed:
            return beams[0][0]

        # Return best sequence
        best_seq = max(
            completed,
            key=lambda x: x[1] / (x[0].size(1) ** length_penalty)
        )[0]

        return best_seq


def load_config(config_path=None):
    if config_path is None:
        script_dir = Path(__file__).parent
        config_path = script_dir.parent / 'config.yaml'

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(checkpoint_path, config, device, model_type='preln'):
    """Load model and tokenizer from checkpoint.

    Args:
        model_type: 'preln' or 'postln' to specify architecture
    """

    tokenizer = MyBPETokenizer(
        texts=["dummy"],
        vocab_size=config['model']['vocab_size'],
        save_dir=config['data']['tokenizer_path']
    )

    vocab_size = len(tokenizer.tokenizer.get_vocab())

    if model_type == 'preln':
        model = PreLNTransformer(
            vocab_size=vocab_size,
            d_model=config['model']['d_model'],
            n_heads=config['model']['n_heads'],
            num_encoder_layers=config['model']['num_encoder_layers'],
            num_decoder_layers=config['model']['num_decoder_layers'],
            dim_feedforward=config['model']['dim_feedforward'],
            max_len=config['model']['max_len'],
            dropout=config['model']['dropout']
        ).to(device)
    else:  # postln
        model = Transformer(
            vocab_size=vocab_size,
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
    print(f"Architecture: {model_type.upper()}")
    print(f"Step: {checkpoint.get('step', 'N/A')}, Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")

    return model, tokenizer


def translate_sentence(model, tokenizer, source_text, device, config):
    model.eval()

    with torch.no_grad():
        src_ids = tokenizer.encode_src(
            source_text,
            max_length=config['data']['max_src_len']
        ).unsqueeze(0).to(device)

        src_mask = (src_ids != tokenizer.pad_id).to(device)

        # Choose decoding method
        method = config['generation'].get('method', 'greedy')

        if method == 'beam':
            output_ids = beam_search_decode(
                model=model,
                src=src_ids,
                src_mask=src_mask,
                max_len=config['generation']['max_length'],
                start_token_id=tokenizer.bos_id,
                end_token_id=tokenizer.eos_id,
                device=device,
                beam_size=config['generation'].get('beam_size', 5),
                length_penalty=config['generation'].get('length_penalty', 0.6)
            )
        else:  
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
