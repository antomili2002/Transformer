import wandb
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8

SINGLE_COL_WIDTH = 3.5
FIG_HEIGHT = 2.5

api = wandb.Api()

postln_runs = api.runs("transformer-translation")
preln_runs = api.runs("transformer-translation-preln")

postln_run = postln_runs[0]
preln_run = preln_runs[0]

print(f"Comparing runs:")
print(f"Post-LN: {postln_run.name}")
print(f"Pre-LN:  {preln_run.name}")

postln_history = postln_run.history()
preln_history = preln_run.history()

# Figure 1: Training Loss
fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, FIG_HEIGHT))
ax.plot(postln_history['step'], postln_history['train_loss'], label='Post-LN', alpha=0.7, linewidth=1.5)
ax.plot(preln_history['step'], preln_history['train_loss'], label='Pre-LN', alpha=0.7, linewidth=1.5)
ax.set_xlabel('Training Step')
ax.set_ylabel('Training Loss')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_train_loss.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_train_loss.png', dpi=300, bbox_inches='tight')
print("Saved fig_train_loss.pdf/png")
plt.close()

# Figure 2: Validation Loss
fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, FIG_HEIGHT))
ax.plot(postln_history['step'], postln_history['val_loss'], label='Post-LN', marker='o', markersize=3, linewidth=1.5)
ax.plot(preln_history['step'], preln_history['val_loss'], label='Pre-LN', marker='s', markersize=3, linewidth=1.5)
ax.set_xlabel('Training Step')
ax.set_ylabel('Validation Loss')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_val_loss.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_val_loss.png', dpi=300, bbox_inches='tight')
print("Saved fig_val_loss.pdf/png")
plt.close()

# Figure 3: BLEU Score
postln_bleu = postln_run.summary.get('test_bleu_score', 0)
preln_bleu = preln_run.summary.get('test_bleu_score', 0)
fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, FIG_HEIGHT))
bars = ax.bar(['Post-LN', 'Pre-LN'], [postln_bleu, preln_bleu], color=['#d62728', '#1f77b4'], width=0.6)
ax.set_ylabel('BLEU Score')
ax.grid(True, axis='y', alpha=0.3)
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('fig_bleu_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_bleu_comparison.png', dpi=300, bbox_inches='tight')
print("Saved fig_bleu_comparison.pdf/png")
plt.close()

# Figure 4: Post-LN Layer Health
layer_cols = [col for col in postln_history.columns if 'decoder/layer_' in col and '_std' in col]
if layer_cols:
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, FIG_HEIGHT))
    for col in sorted(layer_cols):
        layer_num = col.split('_')[1]
        ax.plot(postln_history['step'], postln_history[col], alpha=0.6, label=f'Layer {layer_num}', linewidth=1.2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Standard Deviation')
    ax.axhline(y=0.1, color='r', linestyle='--', linewidth=1.5, label='Collapse Threshold')
    ax.legend(ncol=2, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig_postln_layer_health.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('fig_postln_layer_health.png', dpi=300, bbox_inches='tight')
    print("Saved fig_postln_layer_health.pdf/png")
    plt.close()

    # Figure 5: Pre-LN Layer Health
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, FIG_HEIGHT))
    for col in sorted(layer_cols):
        layer_num = col.split('_')[1]
        ax.plot(preln_history['step'], preln_history[col], alpha=0.6, label=f'Layer {layer_num}', linewidth=1.2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Standard Deviation')
    ax.axhline(y=1.0, color='g', linestyle='--', linewidth=1.5, label='Healthy Threshold')
    ax.legend(ncol=2, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig_preln_layer_health.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('fig_preln_layer_health.png', dpi=300, bbox_inches='tight')
    print("Saved fig_preln_layer_health.pdf/png")
    plt.close()

# Figure 6: Training Speed
if 'time_per_step' in postln_history.columns and 'time_per_step' in preln_history.columns:
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, FIG_HEIGHT))
    ax.plot(postln_history['step'], postln_history['time_per_step'], label='Post-LN', alpha=0.7, linewidth=1.5)
    ax.plot(preln_history['step'], preln_history['time_per_step'], label='Pre-LN', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Time per Step (s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig_training_speed.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('fig_training_speed.png', dpi=300, bbox_inches='tight')
    print("Saved fig_training_speed.pdf/png")
    plt.close()

# Figure 7: Total Time Comparison
postln_time = postln_run.summary.get('total_training_time_hours', 0)
preln_time = preln_run.summary.get('total_training_time_hours', 0)
if postln_time > 0 and preln_time > 0:
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, FIG_HEIGHT))
    bars = ax.bar(['Post-LN', 'Pre-LN'], [postln_time, preln_time], color=['#d62728', '#1f77b4'], width=0.6)
    ax.set_ylabel('Training Time (hours)')
    ax.grid(True, axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}h', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig('fig_total_time.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('fig_total_time.png', dpi=300, bbox_inches='tight')
    print("Saved fig_total_time.pdf/png")
    plt.close()

print("\nAll figures saved!")
