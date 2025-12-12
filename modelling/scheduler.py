"""
Learning rate scheduler from "Attention is All You Need" paper.

The scheduler implements the formula:
    lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))

This schedule increases the learning rate linearly for the first warmup_steps training steps,
and decreases it thereafter proportionally to the inverse square root of the step number.

Reference: Vaswani et al., "Attention is All You Need" (2017)
"""

import torch
from torch.optim.lr_scheduler import LambdaLR


class TransformerScheduler:
    """
    Learning rate scheduler for Transformer models.
    
    Implements the schedule from the "Attention is All You Need" paper:
    lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
    
    This schedule has two phases:
    1. Warmup phase (0 to warmup_steps): Linear increase in learning rate
    2. Decay phase (warmup_steps onwards): Learning rate decays as inverse sqrt of step
    
    Args:
        optimizer: PyTorch optimizer instance
        d_model: Embedding dimension (controls the base learning rate scale)
        warmup_steps: Number of steps to linearly increase learning rate (default: 4000)
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
    def step(self):
        """Update learning rate for the current step and increment step counter."""
        # Calculate learning rate for current step
        lrate = self._compute_lr(self.current_step)
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lrate
        
        self.current_step += 1
        
    def get_lr(self) -> float:
        """Get the current learning rate."""
        return self._compute_lr(self.current_step)
    
    def _compute_lr(self, step: int) -> float:
        """
        Compute learning rate for a given step using the transformer schedule.
        
        Formula: lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
        
        Args:
            step: Current training step (1-indexed for proper behavior)
        
        Returns:
            Learning rate for the given step
        """
        # Avoid division by zero at step 0
        step = max(1, step)
        
        # Compute the two components
        arg1 = step ** (-0.5)
        arg2 = step * (self.warmup_steps ** (-1.5))
        
        # Take minimum (warmup phase vs decay phase)
        lrate = (self.d_model ** (-0.5)) * min(arg1, arg2)
        
        return lrate
    
    def state_dict(self):
        """Return the state of the scheduler."""
        return {'current_step': self.current_step}
    
    def load_state_dict(self, state_dict):
        """Load the state of the scheduler."""
        self.current_step = state_dict['current_step']


class TransformerSchedulerLambda(LambdaLR):
    """
    Alternative implementation using PyTorch's LambdaLR for compatibility.
    
    This is a wrapper around PyTorch's standard LambdaLR that follows
    the learning rate schedule from the Transformer paper.
    
    Args:
        optimizer: PyTorch optimizer instance
        d_model: Embedding dimension
        warmup_steps: Number of warmup steps (default: 4000)
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int = 4000):
        def lr_lambda(step):
            """Lambda function for LambdaLR."""
            step = max(1, step)
            arg1 = step ** (-0.5)
            arg2 = step * (warmup_steps ** (-1.5))
            return (d_model ** (-0.5)) * min(arg1, arg2)
        
        super().__init__(optimizer, lr_lambda)
