# Transformer Learning Rate Scheduler and AdamW Optimizer

## 1. Learning Rate Scheduler

### Overview
The learning rate scheduler from "Attention is All You Need" (Vaswani et al., 2017) implements a schedule that combines:
1. **Warmup phase**: Linear increase in learning rate
2. **Decay phase**: Inverse square root decay

### Mathematical Formula

```
lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
```

Where:
- `d_model`: Embedding dimension (e.g., 512)
- `step_num`: Current training step (1-indexed)
- `warmup_steps`: Number of steps for warmup phase (default: 4000)

### Intuition

**Warmup Phase (step < warmup_steps):**
- The second term dominates: `step_num * warmup_steps^(-1.5)`
- Learning rate increases linearly from near 0 to the base rate
- Purpose: Allow model to stabilize before aggressive updates
- Formula becomes approximately: `lrate ≈ (d_model^(-0.5)) * (step / warmup_steps)`

**Decay Phase (step > warmup_steps):**
- The first term dominates: `step_num^(-0.5)`
- Learning rate decreases proportionally to 1/√step
- Purpose: Fine-tune the model as training progresses
- Slower decay prevents overfitting

### Why This Schedule Works

1. **Empirically effective**: Found to work well in practice without extensive tuning
2. **Theoretically motivated**: 
   - Warmup prevents instability early in training
   - Inverse square root decay provides gentle regularization
3. **Automatic scaling**: Scales with `d_model` so larger models get appropriate rates

### Example Schedule (d_model=512, warmup_steps=4000)

```
Step        Learning Rate    Phase
0           ~1.5e-08        Warmup
1000        ~1.36e-04       Warmup
4000        ~1.56e-04       Peak
8000        ~1.10e-04       Decay
16000       ~7.81e-05       Decay
32000       ~5.52e-05       Decay
```

---

## 2. AdamW Optimizer

### What is AdamW?

AdamW (Adam with decoupled Weight decay) is an improvement over standard Adam that applies weight decay more effectively. 

Key differences from Adam:
- **Decoupled weight decay**: Applied directly to parameters, not via gradients
- **True L2 regularization**: Penalizes parameter magnitude regardless of gradient magnitude
- **Better generalization**: Produces better test performance in practice

### Update Equations

#### Standard Adam with Weight Decay (in gradient):
```
g_t = ∇f(θ_{t-1})  # Gradient
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t  # First moment (momentum)
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²  # Second moment (variance)
m̂_t = m_t / (1 - β₁^t)  # Bias correction
v̂_t = v_t / (1 - β₂^t)  # Bias correction
θ_t = θ_{t-1} - α * (m̂_t / (√v̂_t + ε) + λ * θ_{t-1})  # Weight decay in gradient
```

#### AdamW (Decoupled Weight Decay):
```
g_t = ∇f(θ_{t-1})
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
θ_t = θ_{t-1} - α * (m̂_t / (√v̂_t + ε) + λ * θ_{t-1})  # Weight decay applied separately
```

The key difference: In AdamW, weight decay is applied directly to the parameter, not mixed into the adaptive learning rate term.

### Bias Correction

**Why is bias correction needed?**

At the start of training (small t):
- `m_0 = 0`, `v_0 = 0`
- After first step: `m_1 = (1 - β₁) * g_1` (small!)
- `v_1 = (1 - β₂) * g_1²` (small!)

The estimates are biased toward zero because we start from zero initialization.

**Correction formulas:**
```
m̂_t = m_t / (1 - β₁^t)  # Multiply by (1 - β₁^t)^(-1) to unbias
v̂_t = v_t / (1 - β₂^t)  # Similar for second moment
```

**Example (β₁ = 0.9):**
- At step 1: divide by (1 - 0.9^1) = 0.1 → amplify by 10x
- At step 10: divide by (1 - 0.9^10) ≈ 0.65 → amplify by ~1.5x
- At step 100: divide by (1 - 0.9^100) ≈ 1.0 → no amplification

This ensures reliable estimates throughout training.

---

## 3. Weight Decay Handling in Transformer Training

### Why NOT apply weight decay to all parameters?

Weight decay (L2 regularization) adds a penalty term `λ * θ²` to the loss, effectively shrinking all parameters toward zero.

However, some parameters should NOT be regularized:
1. **Bias parameters** (`b` in Linear layers): Small number of parameters, minimal regularization benefit
2. **Layer normalization parameters** (γ and β): Normalization scale/shift; decaying them doesn't make sense statistically

### Parameter Groups

Our optimizer creates two parameter groups:

```
Group 1: Parameters WITH weight decay
  - Weight matrices in Linear layers
  - Embedding weights
  - Attention projection weights
  
Group 2: Parameters WITHOUT weight decay (λ = 0)
  - Bias terms (any 'bias' parameter)
  - LayerNorm parameters
  - BatchNorm parameters
```

### Implementation in Practice

```python
optimizer = create_adamw_optimizer(
    model,
    learning_rate=1e-4,
    weight_decay=0.01
)
```

This automatically separates parameters based on their names and creates the appropriate parameter groups.

---

## 4. Complete Training Loop Example

```python
import torch
from modelling.model import Transformer
from modelling.optimizer import create_adamw_optimizer
from modelling.scheduler import TransformerScheduler

# Initialize model
model = Transformer(vocab_size=10000, d_model=512, ...)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Initialize optimizer and scheduler
optimizer = create_adamw_optimizer(
    model,
    learning_rate=0.0,  # Will be set by scheduler
    weight_decay=0.01
)
scheduler = TransformerScheduler(optimizer, d_model=512, warmup_steps=4000)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        # Forward pass
        src, tgt, labels = batch
        src = src.to(device)
        tgt = tgt.to(device)
        labels = labels.to(device)
        
        output = model(src, tgt)  # [B, T, vocab_size] after projection
        loss = criterion(output.view(-1, vocab_size), labels.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        scheduler.step()
        
        # Log
        current_lr = scheduler.get_lr()
        print(f"Step {scheduler.current_step}, LR: {current_lr:.6e}, Loss: {loss:.4f}")
```

---

## 5. Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `d_model` | 512 | Embedding dimension; affects base LR scale |
| `warmup_steps` | 4000 | Original paper value; can be tuned |
| `learning_rate` | 1e-4 | Initial peak LR; scheduler overrides |
| `weight_decay` | 0.01 | L2 regularization strength |
| `β₁` (momentum) | 0.9 | Exponential decay for first moment |
| `β₂` (RMSprop) | 0.999 | Exponential decay for second moment |
| `ε` | 1e-8 | Numerical stability in denominator |

---

## References

- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). "Attention is All You Need." *NeurIPS*.
- Loshchilov, I., & Hutter, F. (2019). "Decoupled Weight Decay Regularization." *ICLR*.
- Kingma, D., & Ba, J. (2014). "Adam: A Method for Stochastic Optimization." *ICLR*.
