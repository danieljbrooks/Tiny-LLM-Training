"""Configuration dataclasses for TinyLLM training."""

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    context_length: int = 512
    n_layers: int = 6
    n_heads: int = 6
    d_model: int = 384
    d_ff: Optional[int] = None  # Default: 4 * d_model
    dropout: float = 0.1
    bias: bool = True

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_steps: int = 5000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    eval_interval: int = 1000
    save_interval: int = 1000
    log_interval: int = 100
    device: str = "mps"  # "mps", "cuda", or "cpu"

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    dataset_name: str = "roneneldan/TinyStories"
    tokenizer_name: str = "gpt2"
    data_dir: str = "data"
    train_split: str = "train"
    val_split: str = "validation"
    test_data_size: int = 10000  # Number of samples for validation
    num_workers: int = 0  # For DataLoader (0 recommended for MPS)

    @property
    def train_data_path(self) -> str:
        return f"{self.data_dir}/train.bin"

    @property
    def val_data_path(self) -> str:
        return f"{self.data_dir}/val.bin"


@dataclass
class GenerationConfig:
    """Text generation configuration."""
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.1


# Model size presets
MODEL_PRESETS = {
    "tiny": ModelConfig(
        n_layers=4,
        n_heads=4,
        d_model=256,
        d_ff=1024,
    ),
    "small": ModelConfig(
        n_layers=6,
        n_heads=6,
        d_model=384,
        d_ff=1536,
    ),
    "medium": ModelConfig(
        n_layers=8,
        n_heads=8,
        d_model=512,
        d_ff=2048,
    ),
}


def _default_batch_config() -> tuple[int, int]:
    """Return (batch_size, gradient_accumulation_steps) appropriate for the device."""
    if torch.backends.mps.is_available():
        return 8, 4
    return 32, 1


def get_config(
    model_size: str = "small",
    context_length: int = 512,
    batch_size: int = None,
    learning_rate: float = 3e-4,
    max_steps: int = 5000,
) -> tuple[ModelConfig, TrainingConfig, DataConfig]:
    """Get configuration for a given model size with optional overrides."""
    if model_size not in MODEL_PRESETS:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(MODEL_PRESETS.keys())}")

    model_config = MODEL_PRESETS[model_size]
    model_config.context_length = context_length

    default_bs, default_accum = _default_batch_config()
    if batch_size is None:
        batch_size = default_bs
        grad_accum = default_accum
    else:
        grad_accum = 1

    training_config = TrainingConfig(
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        max_steps=max_steps,
    )

    data_config = DataConfig()

    return model_config, training_config, data_config
