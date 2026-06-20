"""CLI entry point for TinyLLM training and generation."""

import os
import argparse
import torch
import platform


def print_system_info():
    """Print system and PyTorch information."""
    print("\n" + "=" * 80)
    print("System Information")
    print("=" * 80)
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"\nDevice Availability:")
    print(f"  MPS (Apple Silicon): {torch.backends.mps.is_available()}")
    print(f"  CUDA (NVIDIA GPU): {torch.cuda.is_available()}")

    if torch.backends.mps.is_available():
        print(f"\nMPS device will be used for training")
    elif torch.cuda.is_available():
        print(f"\nCUDA device will be used for training")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"\nCPU will be used for training")

    print("=" * 80 + "\n")


def cmd_info(args):
    """Show system information."""
    print_system_info()


def cmd_train(args):
    """Train a model."""
    from config import get_config, DataConfig
    from model import TinyGPT
    from data import get_dataloaders
    from train import Trainer

    # Organize outputs by model name
    log_dir = args.log_dir
    checkpoint_dir = args.checkpoint_dir
    if args.name:
        log_dir = os.path.join(log_dir, args.name)
        checkpoint_dir = os.path.join(checkpoint_dir, args.name)

    # Load model config from checkpoint or CLI args
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, weights_only=False)
        model_config = checkpoint['model_config']
        print(f"\nLoaded model config from checkpoint")
    else:
        model_config, _, _ = get_config(model_size=args.size, context_length=args.context_length)

    # Build training config from CLI args (always use current CLI values)
    _, training_config, _ = get_config(
        model_size=args.size,
        context_length=model_config.context_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
    )

    if args.device:
        training_config.device = args.device

    data_config = DataConfig()

    print("\n" + "=" * 80)
    print(f"TinyLLM Training")
    print("=" * 80)

    print(f"\nModel Configuration:")
    print(f"  Layers: {model_config.n_layers}")
    print(f"  Heads: {model_config.n_heads}")
    print(f"  Dimension: {model_config.d_model}")
    print(f"  FFN Dimension: {model_config.d_ff}")
    print(f"  Context Length: {model_config.context_length}")

    print(f"\nTraining Configuration:")
    print(f"  Batch Size: {training_config.batch_size}")
    print(f"  Gradient Accumulation: {training_config.gradient_accumulation_steps}")
    print(f"  Effective Batch Size: {training_config.effective_batch_size}")
    print(f"  Learning Rate: {training_config.learning_rate}")
    print(f"  Max Steps: {training_config.max_steps}")
    print(f"  Device: {training_config.device}")
    print()

    # Create model
    print("Creating model...")
    model = TinyGPT(model_config)

    # Get dataloaders
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(
        data_config,
        context_length=model_config.context_length,
        batch_size=training_config.batch_size,
        num_workers=data_config.num_workers,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        model_config=model_config,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    print("\nStarting training...")
    trainer.train()

    print(f"\nTraining complete!")
    print(f"Best model saved to: {checkpoint_dir}/best.pt")
    print(f"Logs saved to: {log_dir}")


def cmd_generate(args):
    """Generate stories from a trained model."""
    from config import GenerationConfig
    from generate import StoryGenerator

    print("\n" + "=" * 80)
    print("TinyLLM Story Generation")
    print("=" * 80)

    # Create generation config
    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    # Resolve checkpoint path
    if args.name:
        checkpoint_path = f"checkpoints/{args.name}/best.pt"
    elif args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = "checkpoints/best.pt"

    # Load model
    generator = StoryGenerator.from_checkpoint(
        checkpoint_path,
        device=args.device,
        generation_config=gen_config,
    )

    # Interactive mode or single generation
    if args.interactive:
        generator.interactive_mode()
    else:
        prompt = args.prompt or ""

        print(f"\nPrompt: {prompt if prompt else '(empty)'}")
        print(f"Generating {args.num_samples} sample(s)...\n")

        use_stream = args.num_samples == 1

        stories = generator.generate(
            prompt,
            num_samples=args.num_samples,
            stream=use_stream,
        )

        if not use_stream:
            for i, story in enumerate(stories):
                print(f"\n{'=' * 80}")
                print(f"Sample {i + 1}:")
                print("=" * 80)
                print(story)
                print("=" * 80)


def main():
    """Main CLI entry point."""
    # Set MPS fallback environment variable before anything else
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    parser = argparse.ArgumentParser(
        description="TinyLLM - Train small language models on TinyStories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Model name for organizing checkpoints and logs (e.g. "small-v1")',
    )
    train_parser.add_argument(
        '--size',
        type=str,
        default='small',
        choices=['tiny', 'small', 'medium'],
        help='Model size (default: small)',
    )
    train_parser.add_argument(
        '--context-length',
        type=int,
        default=512,
        help='Context length in tokens (default: 512)',
    )
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (default: 8 on MPS, 32 otherwise)',
    )
    train_parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-4,
        help='Learning rate (default: 3e-4)',
    )
    train_parser.add_argument(
        '--max-steps',
        type=int,
        default=5000,
        help='Number of optimizer steps (default: 5000)',
    )
    train_parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['mps', 'cuda', 'cpu'],
        help='Device to use (default: auto-detect)',
    )
    train_parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory for logs (default: logs)',
    )
    train_parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory for checkpoints (default: checkpoints)',
    )
    train_parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint path',
    )

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate stories')
    gen_parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Model name (loads checkpoints/<name>/best.pt)',
    )
    gen_parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (default: checkpoints/best.pt)',
    )
    gen_parser.add_argument(
        '--prompt',
        type=str,
        default='',
        help='Text prompt (default: empty)',
    )
    gen_parser.add_argument(
        '--max-tokens',
        type=int,
        default=256,
        help='Maximum tokens to generate (default: 256)',
    )
    gen_parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (default: 0.8)',
    )
    gen_parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='Top-k sampling (default: 50)',
    )
    gen_parser.add_argument(
        '--top-p',
        type=float,
        default=0.95,
        help='Top-p (nucleus) sampling (default: 0.95)',
    )
    gen_parser.add_argument(
        '--repetition-penalty',
        type=float,
        default=1.1,
        help='Repetition penalty (default: 1.1)',
    )
    gen_parser.add_argument(
        '--num-samples',
        type=int,
        default=1,
        help='Number of samples to generate (default: 1)',
    )
    gen_parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive generation mode',
    )
    gen_parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['mps', 'cuda', 'cpu'],
        help='Device to use (default: auto-detect)',
    )

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    if args.command == 'info':
        cmd_info(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'generate':
        cmd_generate(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
