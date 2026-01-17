import torch
import torch.nn as nn
from evaluate import load
from typing import Optional, List, Tuple


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


def generate_translation(
    model: nn.Module,
    source_text: str,
    tokenizer,
    max_len: int = 50,
    device: torch.device = torch.device('cpu')
):
    model.eval()

    # tokenize source sentence
    src = tokenizer.encode_src(source_text, max_length=64).unsqueeze(0)  # [1, src_len]

    # Create source mask to ignore padding
    src_mask = (src != tokenizer.pad_id)  # [1, src_len]

    generated = greedy_decode(
        model=model,
        src=src,
        src_mask=src_mask,
        max_len=max_len,
        start_token_id=tokenizer.bos_id,
        end_token_id=tokenizer.eos_id,
        device=device
    )

    token_ids = generated[0].tolist()
    translation = tokenizer.decode(token_ids)

    return translation, token_ids


def evaluate_bleu(
    model: nn.Module,
    test_data: List[dict],
    tokenizer,
    max_len: int = 50,
    device: torch.device = torch.device('cpu'),
    num_samples: Optional[int] = None
):

    bleu = load("bleu")
    model.eval()

    # Limit number of samples if specified
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

        # Generate translation
        translation, _ = generate_translation(
            model, source_text, tokenizer, max_len, device=device
        )

        predictions.append(translation)
        references.append([reference_text])  # BLEU expects list of references

    bleu_results = bleu.compute(predictions=predictions, references=references)

    return {
        'bleu_score': bleu_results['bleu'],
        'predictions': predictions,
        'references': [ref[0] for ref in references],  # Unwrap for easier access
        'source_texts': [sample['src'] for sample in test_data]
    }