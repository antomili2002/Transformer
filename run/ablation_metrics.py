import torch


def compute_layer_health(model, src_ids, src_mask, tokenizer, device):
    """Compute decoder layer std deviation to detect collapse"""
    model.eval()
    with torch.no_grad():
        memory = model.encode(src_ids, src_mask)
        tgt = torch.tensor([[tokenizer.bos_id]], dtype=torch.long, device=device)
        tgt_embedded = model.decoder.embedding(tgt)
        x = tgt_embedded

        metrics = {}
        for i, layer in enumerate(model.decoder.layers):
            x = layer(x, memory, encoder_attention_mask=src_mask, attention_mask=None)
            metrics[f'decoder/layer_{i}_std'] = x.std().item()

    return metrics
