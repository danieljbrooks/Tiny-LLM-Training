"""Dataset loading and preprocessing for TinyStories."""

import os
import numpy as np
from pathlib import Path
from typing import Optional
from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

from config import DataConfig


def preprocess_dataset(
    data_config: DataConfig,
    context_length: int,
    force_reprocess: bool = False,
) -> tuple[str, str]:
    """
    Download and preprocess TinyStories dataset.

    Returns paths to train and validation binary files.
    """
    os.makedirs(data_config.data_dir, exist_ok=True)

    train_path = data_config.train_data_path
    val_path = data_config.val_data_path

    # Check if already preprocessed
    if not force_reprocess and os.path.exists(train_path) and os.path.exists(val_path):
        print(f"Found preprocessed data at {data_config.data_dir}")
        return train_path, val_path

    print("Preprocessing dataset...")
    print(f"Loading {data_config.dataset_name} from HuggingFace...")

    # Load dataset
    dataset = load_dataset(data_config.dataset_name)

    # Load tokenizer
    print(f"Loading tokenizer: {data_config.tokenizer_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(data_config.tokenizer_name)

    # Process train split
    print(f"Processing train split...")
    train_tokens = _tokenize_split(
        dataset[data_config.train_split],
        tokenizer,
        "train",
    )
    _save_tokens(train_tokens, train_path)

    # Process validation split
    print(f"Processing validation split...")
    # Use a subset of train data if no validation split exists
    if data_config.val_split in dataset:
        val_data = dataset[data_config.val_split]
    else:
        # Take last N samples from train for validation
        val_data = dataset[data_config.train_split].select(
            range(len(dataset[data_config.train_split]) - data_config.test_data_size,
                  len(dataset[data_config.train_split]))
        )

    val_tokens = _tokenize_split(val_data, tokenizer, "validation")
    _save_tokens(val_tokens, val_path)

    print(f"Preprocessing complete!")
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")

    return train_path, val_path


def _tokenize_split(data_split, tokenizer, split_name: str) -> np.ndarray:
    """Tokenize a dataset split and return as numpy array."""
    all_tokens = []

    # Process in batches for efficiency
    batch_size = 1000
    for i in tqdm(range(0, len(data_split), batch_size), desc=f"Tokenizing {split_name}"):
        batch = data_split[i:i + batch_size]
        texts = batch["text"]

        # Tokenize batch
        encoded = tokenizer(
            texts,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        # Flatten all token IDs
        for token_ids in encoded["input_ids"]:
            all_tokens.extend(token_ids)

    return np.array(all_tokens, dtype=np.uint16)


def _save_tokens(tokens: np.ndarray, path: str):
    """Save tokens to binary file."""
    tokens.tofile(path)
    print(f"Saved {len(tokens):,} tokens to {path}")


class TinyStoriesDataset(Dataset):
    """Memory-mapped dataset for TinyStories tokens."""

    def __init__(self, data_path: str, context_length: int, stride: Optional[int] = None):
        """
        Args:
            data_path: Path to binary token file
            context_length: Number of tokens per sample
            stride: Stride for sliding window (default: context_length, no overlap)
        """
        self.data_path = data_path
        self.context_length = context_length
        self.stride = stride if stride is not None else context_length

        # Memory-map the token file
        self.tokens = np.memmap(data_path, dtype=np.uint16, mode='r')

        # Calculate number of samples
        self.n_samples = max(1, (len(self.tokens) - context_length) // self.stride + 1)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a training sample.

        Returns dict with 'input_ids' (context) and 'labels' (targets shifted by 1).
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.context_length + 1  # +1 for target

        # Handle edge case at end of dataset
        if end_idx > len(self.tokens):
            end_idx = len(self.tokens)
            start_idx = max(0, end_idx - self.context_length - 1)

        # Get tokens
        chunk = torch.from_numpy(self.tokens[start_idx:end_idx].astype(np.int64))

        # Split into input and target
        x = chunk[:-1]  # All tokens except last
        y = chunk[1:]   # All tokens except first

        return {
            'input_ids': x,
            'labels': y,
        }


def get_dataloaders(
    data_config: DataConfig,
    context_length: int,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        data_config: Data configuration
        context_length: Sequence length
        batch_size: Batch size
        num_workers: Number of dataloader workers (0 recommended for MPS)

    Returns:
        train_loader, val_loader
    """
    # Ensure data is preprocessed
    train_path, val_path = preprocess_dataset(data_config, context_length)

    # Create datasets with some overlap for training
    train_dataset = TinyStoriesDataset(
        train_path,
        context_length=context_length,
        stride=context_length // 2,  # 50% overlap
    )

    val_dataset = TinyStoriesDataset(
        val_path,
        context_length=context_length,
        stride=context_length,  # No overlap for validation
    )

    print(f"Train dataset: {len(train_dataset):,} samples")
    print(f"Val dataset: {len(val_dataset):,} samples")

    # Create dataloaders (pin_memory=False for MPS)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader
