"""Training loop with logging and checkpointing."""

import os
import csv
import json
import time
import itertools
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import ModelConfig, TrainingConfig
from model import TinyGPT


class Trainer:
    """Handles model training with logging and checkpointing."""

    def __init__(
        self,
        model: TinyGPT,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: TrainingConfig,
        model_config: ModelConfig,
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.model_config = model_config

        # Setup directories
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup device
        self.device = self._setup_device(config.device)
        self.model = self.model.to(self.device)
        if hasattr(torch, 'compile') and self.device.type != 'mps':
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass
        print(f"Using device: {self.device}")

        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        # Cosine schedule with warmup
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, config.max_steps - config.warmup_steps),
            eta_min=config.learning_rate * 0.1,
        )

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.start_time = None

        # Setup logging
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"train_{timestamp}.log"
        self.loss_csv = self.log_dir / f"losses_{timestamp}.csv"
        with open(self.loss_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["step", "train_loss", "val_loss", "lr", "elapsed_sec"])

    def _setup_device(self, device: str) -> torch.device:
        """Setup and configure device for training."""
        if device == "mps" and torch.backends.mps.is_available():
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            return torch.device("mps")
        elif device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("Warning: Requested device not available, falling back to CPU")
            return torch.device("cpu")

    def _log(self, message: str):
        """Log message to file and console."""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

    def _log_loss(self, step: int, train_loss: float, val_loss: float = None):
        elapsed = time.time() - self.start_time if self.start_time else 0
        lr = self.optimizer.param_groups[0]['lr']
        with open(self.loss_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, f"{train_loss:.6f}", f"{val_loss:.6f}" if val_loss is not None else "", f"{lr:.2e}", f"{elapsed:.1f}"])

    def _warmup_lr(self, step: int):
        """Linear warmup for learning rate."""
        if step < self.config.warmup_steps:
            lr_scale = (step + 1) / self.config.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * lr_scale

    def _infinite_batches(self):
        while True:
            yield from self.train_loader

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False, position=0):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            logits, loss = self.model(input_ids, labels)
            total_loss += loss.item()
            num_batches += 1

        if self.device.type == "mps":
            torch.mps.synchronize()

        return total_loss / num_batches

    def train(self):
        """Full training loop."""
        max_steps = self.config.max_steps

        self._log("=" * 80)
        self._log(f"Training for {max_steps:,} steps")
        self._log(f"Model: {self.model.get_num_params():,} parameters")
        self._log(f"Effective batch size: {self.config.effective_batch_size}")
        self._log("=" * 80)

        self.start_time = time.time()
        self.model.train()

        running_loss = 0.0
        micro_step = 0
        progress_bar = tqdm(total=max_steps, desc="Training", initial=self.global_step)

        try:
            for batch in self._infinite_batches():
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits, loss = self.model(input_ids, labels)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                running_loss += loss.item() * self.config.gradient_accumulation_steps
                micro_step += 1

                if micro_step % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if self.global_step < self.config.warmup_steps:
                        self._warmup_lr(self.global_step)
                    else:
                        self.scheduler.step()

                    self.global_step += 1
                    avg_loss = running_loss / self.config.gradient_accumulation_steps
                    running_loss = 0.0

                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'loss': f"{avg_loss:.4f}",
                        'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    })

                    if self.global_step % self.config.log_interval == 0:
                        self._log(
                            f"Step {self.global_step}: "
                            f"loss={avg_loss:.4f}, "
                            f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                        )
                        self._log_loss(self.global_step, avg_loss)

                    if self.global_step % self.config.eval_interval == 0:
                        val_loss = self.evaluate()
                        self._log(f"Step {self.global_step}: val_loss={val_loss:.4f}")
                        self._log_loss(self.global_step, avg_loss, val_loss=val_loss)

                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint("best.pt")
                            self._log(f"New best validation loss: {val_loss:.4f}")

                        self.model.train()

                    if self.global_step % self.config.save_interval == 0:
                        self.save_checkpoint(f"step_{self.global_step}.pt")

                    if self.device.type == "mps" and self.global_step % 50 == 0:
                        torch.mps.empty_cache()

                    if self.global_step >= max_steps:
                        break

        except KeyboardInterrupt:
            self._log("\nTraining interrupted by user")

        progress_bar.close()

        # Final eval and save
        val_loss = self.evaluate()
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint("best.pt")
        self.save_checkpoint("final.pt")
        self._save_summary()

        total_time = time.time() - self.start_time
        self._log(f"\nTraining complete! Total time: {total_time / 60:.1f} minutes")
        self._log(f"Best validation loss: {self.best_val_loss:.4f}")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'model_config': self.model_config,
            'training_config': self.config,
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint and rebuild scheduler for current max_steps."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        # Rebuild scheduler for the new max_steps and fast-forward to current step
        remaining = max(1, self.config.max_steps - self.config.warmup_steps)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=remaining,
            eta_min=self.config.learning_rate * 0.1,
        )
        for _ in range(max(0, self.global_step - self.config.warmup_steps)):
            self.scheduler.step()

        self._log(f"Resumed from {path} at step {self.global_step}")
        self._log(f"Continuing to step {self.config.max_steps}")

    def _save_summary(self):
        """Save training summary as JSON."""
        summary = {
            'model_config': {
                'n_layers': self.model_config.n_layers,
                'n_heads': self.model_config.n_heads,
                'd_model': self.model_config.d_model,
                'd_ff': self.model_config.d_ff,
                'context_length': self.model_config.context_length,
                'num_params': self.model.get_num_params(),
            },
            'training_config': {
                'batch_size': self.config.batch_size,
                'gradient_accumulation_steps': self.config.gradient_accumulation_steps,
                'effective_batch_size': self.config.effective_batch_size,
                'learning_rate': self.config.learning_rate,
                'max_steps': self.config.max_steps,
            },
            'results': {
                'best_val_loss': self.best_val_loss,
                'steps_completed': self.global_step,
                'total_time_seconds': time.time() - self.start_time,
                'total_time_minutes': (time.time() - self.start_time) / 60,
            },
            'device': str(self.device),
        }

        summary_path = self.log_dir / f"summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        self._log(f"Saved training summary to {summary_path}")
