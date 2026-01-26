import os
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
preln_run = preln_runs[4]

print(f"\nComparing runs:")
print(f"Post-LN: {postln_run.name}")
print(f"Pre-LN:  {preln_run.name}")

postln_history = postln_run.history()
preln_history = preln_run.history()

save_dir = "./figures/"
os.makedirs(save_dir, exist_ok=True)

# Training Loss
fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, FIG_HEIGHT))
ax.plot(postln_history['step'], postln_history['train_loss'], label='Post-LN', alpha=0.7, linewidth=1.5)
ax.plot(preln_history['step'], preln_history['train_loss'], label='Pre-LN', alpha=0.7, linewidth=1.5)
ax.set_xlabel('Training Step')
ax.set_ylabel('Training Loss')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{save_dir}fig_train_loss.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{save_dir}fig_train_loss.png', dpi=300, bbox_inches='tight')
print("Saved fig_train_loss.pdf/png")
plt.close()

# Validation Loss
fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, FIG_HEIGHT))
postln_val = postln_history[['step', 'val_loss']].dropna()
preln_val = preln_history[['step', 'val_loss']].dropna()
ax.plot(postln_val['step'], postln_val['val_loss'], label='Post-LN', marker='o', markersize=4, linewidth=2, linestyle='-')
ax.plot(preln_val['step'], preln_val['val_loss'], label='Pre-LN', marker='s', markersize=4, linewidth=2, linestyle='-')
ax.set_xlabel('Training Step')
ax.set_ylabel('Validation Loss')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{save_dir}fig_val_loss.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{save_dir}fig_val_loss.png', dpi=300, bbox_inches='tight')
print("Saved fig_val_loss.pdf/png")
plt.close()

# BLEU Score
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
plt.savefig(f'{save_dir}fig_bleu_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{save_dir}fig_bleu_comparison.png', dpi=300, bbox_inches='tight')
print("Saved fig_bleu_comparison.pdf/png")
plt.close()


# training Speed
if 'time_per_step' in postln_history.columns and 'time_per_step' in preln_history.columns:
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, FIG_HEIGHT))
    ax.plot(postln_history['step'], postln_history['time_per_step'], label='Post-LN', alpha=0.7, linewidth=1.5)
    ax.plot(preln_history['step'], preln_history['time_per_step'], label='Pre-LN', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Time per Step (s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}fig_training_speed.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}fig_training_speed.png', dpi=300, bbox_inches='tight')
    print("Saved fig_training_speed.pdf/png")
    plt.close()

# Total Time Comparison
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
    plt.savefig(f'{save_dir}fig_total_time.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}fig_total_time.png', dpi=300, bbox_inches='tight')
    print("Saved fig_total_time.pdf/png")
    plt.close()

print("\nAll figures saved!")
