# Tiny-LLM-Training

Minimal PyTorch implementation for training small language models (1M-33M parameters) on the TinyStories dataset, optimized for M2 MacBook with MPS acceleration.

See [results.md](results.md) for training curves and example story continuations.

## Features

- **GPT-style decoder-only transformer** with pre-norm layers
- **Weight tying** between input embeddings and output projection
- **Three model sizes**:
  - `tiny`: 4 layers, 4 heads, 256 dim (~10M params)
  - `small`: 6 layers, 6 heads, 384 dim (~22M params)
  - `medium`: 8 layers, 8 heads, 512 dim (~33M params)
- **MPS-optimized** for Apple Silicon with automatic fallback
- **Memory-mapped data loading** for efficient dataset access
- **Comprehensive logging** with JSON training summaries
- **Flexible generation** with temperature, top-k, top-p, and repetition penalty

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- transformers
- datasets
- numpy
- tqdm

## Quick Start

### 1. Check System Information

```bash
python main.py info
```

This will show your system configuration and verify MPS/CUDA availability.

### 2. Train a Model

```bash
# Train small model for 5000 steps (default)
python main.py train --name small-v1 --size small

# Quick test run
python main.py train --name test-run --size tiny --max-steps 500

# Longer training with custom settings
python main.py train --name small-1024ctx --size small --max-steps 20000 --context-length 1024 --batch-size 16
```

The `--name` flag organizes checkpoints and logs into subdirectories (e.g. `checkpoints/small-v1/best.pt`, `logs/small-v1/`), making it easy to manage multiple experiments. Without `--name`, files are saved directly in `checkpoints/` and `logs/`.

Training is step-based rather than epoch-based — the data iterator cycles automatically. The default of 5000 steps takes roughly 40 minutes on an M2 MacBook.

For longer training runs on macOS, use `caffeinate` to prevent the system from sleeping:

```bash
caffeinate -ims python main.py train --name small-v1 --size small --max-steps 20000
```

The first training run will automatically download the TinyStories dataset from HuggingFace and preprocess it into binary files in the `data/` directory.

### 3. Generate Stories

```bash
# Generate from a named model
python main.py generate --name small-v1 --prompt "Once upon a time"

# Or specify a checkpoint path directly
python main.py generate --checkpoint checkpoints/small-v1/best.pt --prompt "Once upon a time"

# Interactive mode
python main.py generate --name small-v1 --interactive

# Multiple samples with custom settings
python main.py generate --name small-v1 \
    --prompt "A little girl" \
    --num-samples 3 \
    --temperature 0.9 \
    --top-k 50 \
    --top-p 0.95
```

## Project Structure

```
Tiny-LLM-Benchmarks/
├── config.py          # Hyperparameter dataclasses and model presets
├── data.py            # Dataset loading & preprocessing
├── model.py           # GPT-style transformer implementation
├── train.py           # Training loop with logging & checkpointing
├── generate.py        # Inference and story generation
├── main.py            # CLI entry point
├── requirements.txt   # Dependencies
├── data/              # Preprocessed binary tokens (gitignored)
├── checkpoints/       # Model checkpoints, organized by --name (gitignored)
└── logs/              # Training logs with JSON summaries, organized by --name (gitignored)
```

## Model Architecture

- **Architecture**: GPT-style decoder-only transformer
- **Normalization**: Pre-norm with LayerNorm
- **Activation**: GELU
- **Dropout**: 0.1
- **Context Length**: 512 tokens (configurable, 256-1024 recommended)
- **Tokenizer**: GPT-2 tokenizer (50,257 vocab size)

### Model Sizes

| Size   | Layers | Heads | d_model | d_ff  | Parameters |
|--------|--------|-------|---------|-------|------------|
| tiny   | 4      | 4     | 256     | 1024  | ~10M       |
| small  | 6      | 6     | 384     | 1536  | ~22M       |
| medium | 8      | 8     | 512     | 2048  | ~33M       |

## Training Configuration

### Default Settings
- **Optimizer**: AdamW (lr=3e-4, weight_decay=0.1)
- **Scheduler**: Cosine annealing with linear warmup (100 steps)
- **Batch Size**: 8 on MPS, 32 otherwise (configurable)
- **Gradient Accumulation**: 4 on MPS, 1 otherwise (effective batch size: 32)
- **Gradient Clipping**: Max norm 1.0
- **Max Steps**: 5000 (configurable)

### MPS Optimizations
- `PYTORCH_ENABLE_MPS_FALLBACK=1` automatically set
- `pin_memory=False` in DataLoaders
- Periodic `torch.mps.empty_cache()` calls
- `torch.mps.synchronize()` for accurate timing
- Float32 only (no mixed precision)

## Command Line Interface

### Train Command

```bash
python main.py train [OPTIONS]

Options:
  --name TEXT                    Model name for organizing outputs (e.g. "small-v1")
  --size {tiny,small,medium}     Model size (default: small)
  --context-length INT           Context length in tokens (default: 512)
  --batch-size INT               Batch size (default: 8 on MPS, 32 otherwise)
  --learning-rate FLOAT          Learning rate (default: 3e-4)
  --max-steps INT                Number of optimizer steps (default: 5000)
  --device {mps,cuda,cpu}        Device to use (default: auto-detect)
  --log-dir PATH                 Directory for logs (default: logs)
  --checkpoint-dir PATH          Directory for checkpoints (default: checkpoints)
  --resume PATH                  Resume from checkpoint path
```

### Generate Command

```bash
python main.py generate [OPTIONS]

Options:
  --name TEXT                    Model name (loads checkpoints/<name>/best.pt)
  --checkpoint PATH              Path to model checkpoint (default: checkpoints/best.pt)
  --prompt TEXT                  Text prompt (default: empty)
  --max-tokens INT               Maximum tokens to generate (default: 256)
  --temperature FLOAT            Sampling temperature (default: 0.8)
  --top-k INT                    Top-k sampling (default: 50)
  --top-p FLOAT                  Top-p (nucleus) sampling (default: 0.95)
  --repetition-penalty FLOAT     Repetition penalty (default: 1.1)
  --num-samples INT              Number of samples to generate (default: 1)
  --interactive                  Interactive generation mode
  --device {mps,cuda,cpu}        Device to use (default: auto-detect)
```

### Info Command

```bash
python main.py info
```

Shows system information and device availability.

## Dataset

This project uses the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset, a collection of short stories generated with GPT-3.5/GPT-4, designed for training small language models.

The dataset is automatically downloaded and preprocessed on first use:
- Tokenized using GPT-2 tokenizer
- Saved as memory-mapped binary files (uint16)
- Training samples use sliding window with 50% overlap
- Validation samples use non-overlapping windows

## Logging and Checkpointing

### Training Logs
- Console output with progress bars
- Text log files in `logs/` directory
- JSON summaries with model config, training config, and results

### Checkpoints
- `best.pt`: Best model based on validation loss
- `final.pt`: Final model after training
- `step_N.pt`: Periodic checkpoints during training

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Global step count
- Best validation loss
- Model and training configurations

## License

MIT License - see LICENSE file for details

## Acknowledgments

- TinyStories dataset: [Paper](https://arxiv.org/abs/2305.07759)
- Architecture inspired by nanoGPT by Andrej Karpathy
