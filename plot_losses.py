"""Plot training and validation loss curves for all models."""

import csv
import matplotlib.pyplot as plt
from pathlib import Path


def load_losses(csv_path):
    steps, train_loss, val_steps, val_loss = [], [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row['step'])
            steps.append(step)
            train_loss.append(float(row['train_loss']))
            if row['val_loss']:
                val_steps.append(step)
                val_loss.append(float(row['val_loss']))
    return steps, train_loss, val_steps, val_loss


def get_lr_label(csv_path):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        row = next(reader)
        return row['lr']


MODELS = [
    ("tiny-v2", "logs/tiny-v2/losses_20260618_214612.csv"),
    ("tiny-v3", "logs/tiny-v3/losses_20260619_074034.csv"),
    ("tiny-v4", "logs/tiny-v4/losses_20260619_135310.csv"),
]

fig, ax = plt.subplots(figsize=(10, 6))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for (name, csv_path), color in zip(MODELS, colors):
    lr = get_lr_label(csv_path)
    label = f"lr={lr}"
    steps, train_loss, val_steps, val_loss = load_losses(csv_path)

    ax.plot(steps, train_loss, color=color, alpha=0.7, label=f"{label} (train)")
    if val_loss:
        ax.plot(val_steps, val_loss, color=color, linestyle='--', marker='o',
                markersize=4, label=f"{label} (val)")

ax.set_xlabel("Training Steps")
ax.set_ylabel("Loss")
ax.set_title("TinyLLM Training Loss Curves")
ax.legend()
ax.set_xlim(0, 10000)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("logs/loss_curves.png", dpi=150)
print("Saved to logs/loss_curves.png")
plt.show()
