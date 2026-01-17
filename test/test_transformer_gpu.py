"""Test Transformer model for correctness and GPU compatibility."""
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelling.model import Transformer


def test_transformer_cpu():
    """Test Transformer on CPU."""
    print("Testing Transformer on CPU...")
    
    # Create a small transformer
    vocab_size = 1000
    d_model = 256
    batch_size = 2
    seq_len = 10
    
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        max_len=512,
        dim_feedforward=512,
        dropout=0.1
    )
    
    # Create dummy input
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    output = model(src, tgt)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print(f"CPU test passed. Output shape: {output.shape}")


def test_transformer_gpu():
    """Test Transformer on GPU if available."""
    if not torch.cuda.is_available():
        print("GPU not available, skipping GPU test")
        return
    
    print("\nTesting Transformer on GPU...")
    device = torch.device("cuda")
    
    # Create a small transformer
    vocab_size = 1000
    d_model = 256
    batch_size = 2
    seq_len = 10
    
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        max_len=512,
        dim_feedforward=512,
        dropout=0.1
    )
    
    # Move model to GPU
    model = model.to(device)
    
    # Create dummy input on GPU
    src = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Forward pass
    output = model(src, tgt)
    
    # Check output shape and device
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    assert output.device.type == "cuda", f"Expected output on GPU, got {output.device}"
    print(f"GPU test passed. Output shape: {output.shape}, Device: {output.device}")


def test_transformer_with_masks():
    """Test Transformer with attention masks."""
    print("Testing Transformer with attention masks...")
    
    vocab_size = 1000
    d_model = 256
    batch_size = 2
    seq_len = 10
    
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        max_len=512,
        dim_feedforward=512,
        dropout=0.1
    )
    
    # Create dummy input
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create masks (1=keep, 0=pad)
    src_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    src_mask[0, 8:] = 0  # Mask last 2 tokens of first sample
    
    tgt_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    tgt_mask[1, 7:] = 0  # Mask last 3 tokens of second sample
    
    # Forward pass with masks
    output = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print(f"Mask test passed. Output shape: {output.shape}")


if __name__ == "__main__":
    test_transformer_cpu()
    #test_transformer_gpu()
    test_transformer_with_masks()
    print("All tests passed!")
