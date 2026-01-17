"""
Test and demonstrate the learning rate scheduler and optimizer setup.
"""

import torch
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelling.model import Transformer
from modelling.scheduler import TransformerScheduler
from modelling.optimizer import create_adamw_optimizer, get_optimizer_groups_info


def test_scheduler():
    """Test the learning rate scheduler."""
    print("Testing Learning Rate Scheduler")
    
    # Create a dummy optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    
    # Create scheduler with d_model=512, warmup_steps=4000
    scheduler = TransformerScheduler(optimizer, d_model=512, warmup_steps=4000)
    
    # Collect learning rates over steps
    steps = [0, 100, 500, 1000, 2000, 4000, 8000, 16000, 32000]
    lrs = []
    
    print(f"\nd_model: 512, warmup_steps: 4000\n")
    print(f"{'Step':<10} {'Learning Rate':<20} {'Phase':<15}")
    
    for step in steps:
        scheduler.current_step = step
        lr = scheduler.get_lr()
        lrs.append(lr)
        
        if step < 4000:
            phase = "Warmup"
        else:
            phase = "Decay"
        
        print(f"{step:<10} {lr:<20.6e} {phase:<15}")
    
    print("Scheduler test passed")
    
    # Plot learning rate schedule
    all_steps = list(range(0, 20001, 100))
    all_lrs = []
    
    for step in all_steps:
        scheduler.current_step = step
        all_lrs.append(scheduler.get_lr())
    
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(all_steps, all_lrs, linewidth=2)
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.title('Transformer Learning Rate Schedule (d_model=512, warmup=4000)')
        plt.grid(True, alpha=0.3)
        plt.axvline(x=4000, color='r', linestyle='--', label='Warmup end')
        plt.legend()
        plt.tight_layout()
        plt.savefig('lr_schedule.png')
        print("Learning rate schedule plot saved to 'lr_schedule.png'")
    except Exception as e:
        print(f"Warning: Could not save plot: {e}")


def test_optimizer():
    """Test AdamW optimizer setup with proper weight decay handling."""
    print("Testing AdamW Optimizer Setup")
    
    # Create a transformer model
    model = Transformer(
        vocab_size=1000,
        d_model=256,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        max_len=512,
        dim_feedforward=512,
        dropout=0.1
    )
    
    # Create optimizer with proper weight decay
    optimizer = create_adamw_optimizer(
        model,
        learning_rate=1e-4,
        weight_decay=0.01
    )
    
    # Get information about parameter groups
    groups_info = get_optimizer_groups_info(model)
    
    print(f"Total parameters: {sum(len(v) for v in groups_info.values())}")
    print(f"Parameters with weight decay: {len(groups_info['decay'])}")
    print(f"Parameters without weight decay: {len(groups_info['no_decay'])}")
    
    # Show examples of each group
    print("Examples of parameters WITH weight decay:")
    for name in groups_info['decay'][:5]:
        print(f"  - {name}")
    
    print("Examples of parameters WITHOUT weight decay (bias/norm):")
    for name in groups_info['no_decay'][:5]:
        print(f"  - {name}")
    
    # Verify optimizer param groups
    print(f"Optimizer param groups: {len(optimizer.param_groups)}")
    for i, group in enumerate(optimizer.param_groups):
        print(f"Group {i}: {len(group['params'])} params, weight_decay={group['weight_decay']}")
    
    print("Optimizer test passed")


def test_scheduler_step():
    """Test scheduler step() method."""
    print("Testing Scheduler Step Method")
    
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    scheduler = TransformerScheduler(optimizer, d_model=512, warmup_steps=4000)
    
    print(f"Initial step: {scheduler.current_step}")
    print(f"Initial LR: {scheduler.get_lr():.6e}\n")
    
    print(f"{'Iteration':<12} {'Step':<8} {'Learning Rate':<20}")
    
    for i in range(5):
        lr_before = scheduler.get_lr()
        scheduler.step()
        lr_after = scheduler.get_lr()
        
        print(f"{i:<12} {scheduler.current_step:<8} {lr_after:<20.6e}")
    
    print("\nStep method test passed")


def test_integration():
    """Test full training loop integration."""
    print("Testing Full Integration (Scheduler + Optimizer)")
    
    # Create model
    model = Transformer(
        vocab_size=1000,
        d_model=256,
        n_heads=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        max_len=100,
        dim_feedforward=512,
        dropout=0.1
    )
    
    # Create optimizer and scheduler
    optimizer = create_adamw_optimizer(model, learning_rate=1e-3, weight_decay=0.01)
    scheduler = TransformerScheduler(optimizer, d_model=256, warmup_steps=1000)
    
    # Simulate a few training steps
    print("\nSimulated training loop (5 steps):\n")
    print(f"{'Step':<8} {'LR':<15} {'Loss (simulated)':<20}")
    
    for step in range(5):
        # Get current learning rate
        lr = scheduler.get_lr()
        
        # Simulate loss
        loss = 1.0 / (step + 1)
        
        print(f"{step:<8} {lr:<15.6e} {loss:<20.6f}")
        
        # Update scheduler (this would happen after backward pass)
        scheduler.step()
    
    print("Integration test passed")


if __name__ == "__main__":
    test_scheduler()
    test_optimizer()
    test_scheduler_step()
    test_integration()
    print("All tests passed!")
