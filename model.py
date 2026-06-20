"""GPT-style transformer model implementation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with flash attention support."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.dropout_p = config.dropout

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """Small GPT-style language model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # Output projection (weight tied with token embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.token_embedding.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Apply special scaled init to residual projections
        for name, param in self.named_parameters():
            if name.endswith('proj.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

        print(f"Model initialized with {self.get_num_params():,} parameters")

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: (B, T) tensor of token indices
            labels: (B, T) tensor of target token indices (for training)

        Returns:
            logits: (B, T, vocab_size) tensor of logits
            loss: scalar loss if labels provided, else None
        """
        B, T = input_ids.shape
        assert T <= self.config.context_length, f"Sequence length {T} exceeds max {self.config.context_length}"

        # Get embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)  # (T,)
        tok_emb = self.token_embedding(input_ids)  # (B, T, d_model)
        pos_emb = self.position_embedding(pos)     # (T, d_model)
        x = self.dropout(tok_emb + pos_emb)        # (B, T, d_model)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm and projection
        x = self.ln_f(x)

        if labels is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-1,
            )
            return logits, loss

        logits = self.lm_head(x)
        return logits, None

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: (B, T) tensor of initial tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens (nucleus filtering)
            top_p: Keep top tokens with cumulative probability >= top_p
            repetition_penalty: Penalty for repeating tokens (> 1.0 discourages repetition)

        Returns:
            (B, T + max_new_tokens) tensor of generated tokens
        """
        for _ in range(max_new_tokens):
            input_ids_crop = input_ids if input_ids.size(1) <= self.config.context_length else input_ids[:, -self.config.context_length:]

            logits, _ = self(input_ids_crop)
            # Move to CPU for sampling ops (avoids MPS fallback overhead)
            logits = logits[:, -1, :].cpu() / temperature

            if repetition_penalty != 1.0:
                for i in range(input_ids.size(0)):
                    for token_id in set(input_ids[i].tolist()):
                        logits[i, token_id] /= repetition_penalty

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                for i in range(logits.size(0)):
                    indices_to_remove = sorted_indices[i, sorted_indices_to_remove[i]]
                    logits[i, indices_to_remove] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).to(input_ids.device)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
