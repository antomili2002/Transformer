"""
AdamW Optimizer setup for Transformer models.

This module provides utilities to create an AdamW optimizer with proper weight decay
handling. Weight decay is NOT applied to bias and layer normalization parameters,
following best practices from the literature.

Reference: Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2019)
"""

import torch
import torch.nn as nn
from typing import Dict, List


def create_adamw_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-3,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01
) -> torch.optim.AdamW:
    """
    Create an AdamW optimizer with proper weight decay handling.
    
    Weight decay is NOT applied to:
    - Bias parameters (e.g., in Linear layers)
    - Layer normalization parameters (gamma and beta in LayerNorm)
    
    This follows the principle that weight decay should regularize the main
    weight matrices, not auxiliary parameters like biases and normalization scales.
    
    Args:
        model: PyTorch model to optimize
        learning_rate: Initial learning rate (default: 1e-3)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Small constant for numerical stability (default: 1e-8)
        weight_decay: L2 regularization coefficient (default: 0.01)
    
    Returns:
        AdamW optimizer with proper parameter groups
    
    Example:
        >>> model = Transformer(vocab_size=10000, d_model=512)
        >>> optimizer = create_adamw_optimizer(model, learning_rate=1e-4, weight_decay=0.01)
        >>> from modelling.scheduler import TransformerScheduler
        >>> scheduler = TransformerScheduler(optimizer, d_model=512, warmup_steps=4000)
    """
    
    # Separate parameters into decay and no-decay groups
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Check if this parameter should have weight decay
            if _should_have_decay(name):
                decay_params.append(param)
            else:
                no_decay_params.append(param)
    
    # Create parameter groups with different weight decays
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=learning_rate,
        betas=betas,
        eps=eps
    )
    
    return optimizer


def _should_have_decay(param_name: str) -> bool:
    """
    Determine if a parameter should have weight decay applied.
    
    Returns False for:
    - Bias parameters (any param with 'bias' in the name)
    - Layer normalization parameters ('gamma', 'beta', 'weight' in LayerNorm/BatchNorm)
    
    Args:
        param_name: Name of the parameter (from model.named_parameters())
    
    Returns:
        True if weight decay should be applied, False otherwise
    """
    # Never decay biases
    if 'bias' in param_name:
        return False
    
    # Never decay layer normalization parameters
    # LayerNorm, BatchNorm, GroupNorm all use gamma/beta or weight/bias for scale/shift
    if 'norm' in param_name.lower():
        return False
    
    # Decay all other parameters (main weight matrices)
    return True


def get_optimizer_groups_info(model: nn.Module) -> Dict[str, List[str]]:
    """
    Get information about which parameters are in which optimizer groups.
    
    Useful for debugging and verifying that weight decay is applied correctly.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with keys 'decay' and 'no_decay', each mapping to list of param names
    """
    decay_names = []
    no_decay_names = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if _should_have_decay(name):
                decay_names.append(name)
            else:
                no_decay_names.append(name)
    
    return {
        'decay': decay_names,
        'no_decay': no_decay_names
    }
