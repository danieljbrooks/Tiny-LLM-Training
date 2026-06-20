"""Story generation and inference."""

import sys
import torch
from transformers import GPT2Tokenizer
from typing import Optional

from model import TinyGPT
from config import GenerationConfig, ModelConfig


class StoryGenerator:
    """Handles text generation from trained models."""

    def __init__(
        self,
        model: TinyGPT,
        tokenizer: GPT2Tokenizer,
        device: torch.device,
        generation_config: Optional[GenerationConfig] = None,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.config = generation_config or GenerationConfig()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> 'StoryGenerator':
        """Load model from checkpoint."""
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        device = torch.device(device)
        print(f"Loading model on {device}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if isinstance(checkpoint.get('model_config'), ModelConfig):
            model_config = checkpoint['model_config']
        else:
            raise ValueError("Model config not found in checkpoint")

        model = TinyGPT(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        print(f"Loaded model with {model.get_num_params():,} parameters")

        return cls(model, tokenizer, device, generation_config)

    @torch.no_grad()
    def generate(
        self,
        prompt: str = "",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        num_samples: int = 1,
        stream: bool = False,
    ) -> list[str]:
        """Generate text from a prompt."""
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        top_k = top_k or self.config.top_k
        top_p = top_p or self.config.top_p
        repetition_penalty = repetition_penalty or self.config.repetition_penalty

        if prompt:
            token_ids = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        else:
            input_ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)

        if num_samples > 1:
            input_ids = input_ids.repeat(num_samples, 1)

        if stream and num_samples == 1:
            return [self._generate_streaming(input_ids, max_new_tokens, temperature, top_k, top_p, repetition_penalty)]

        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        outputs = []
        for i in range(num_samples):
            text = self.tokenizer.decode(output_ids[i].tolist(), skip_special_tokens=True)
            outputs.append(text)

        return outputs

    def _generate_streaming(self, input_ids, max_new_tokens, temperature, top_k, top_p, repetition_penalty) -> str:
        """Generate tokens one at a time, printing as they are produced."""
        generated_text = ""

        for _ in range(max_new_tokens):
            input_ids_crop = input_ids if input_ids.size(1) <= self.model.config.context_length else input_ids[:, -self.model.config.context_length:]

            logits, _ = self.model(input_ids_crop)
            logits = logits[:, -1, :].cpu() / temperature

            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    logits[0, token_id] /= repetition_penalty

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices[0, sorted_indices_to_remove[0]]
                logits[0, indices_to_remove] = -float('inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).to(input_ids.device)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            token_text = self.tokenizer.decode([next_token.item()])
            generated_text += token_text

            sys.stdout.write(token_text)
            sys.stdout.flush()

        sys.stdout.write("\n")
        return generated_text

    def interactive_mode(self):
        """Interactive generation mode."""
        print("\n" + "=" * 80)
        print("Interactive Story Generation")
        print("=" * 80)
        print("Enter a prompt (or 'quit' to exit)")
        print(f"Settings: temp={self.config.temperature}, top_k={self.config.top_k}, "
              f"top_p={self.config.top_p}, max_tokens={self.config.max_new_tokens}")
        print("=" * 80 + "\n")

        while True:
            prompt = input("\nPrompt: ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not prompt:
                prompt = ""

            print()
            self.generate(prompt, num_samples=1, stream=True)
