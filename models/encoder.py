"""
Causal Transformer encoder used as the sequence backbone of Agent_S in the
batched-time MAKT system (model.py, MAKTSystem).

The encoder processes the per-step agent messages in parallel across time
while preserving causality: position t can only attend to positions <= t.
Padding positions are excluded through the key padding mask.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CausalTransformerLayer(nn.Module):
    """Pre-norm Transformer layer with a causal self-attention mask."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x    : input features, (B, T, D)
            mask : validity mask, (B, T) with 1 for real positions
        Returns:
            encoded features, (B, T, D)
        """
        seq_len = x.size(1)

        # Upper-triangular True entries are disallowed (future positions).
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()
        key_padding_mask = ~mask.bool() if mask is not None else None

        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class CausalTransformerEncoder(nn.Module):
    """Stack of causal Transformer layers followed by a final LayerNorm."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            CausalTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x    : input features, (B, T, D)
            mask : validity mask, (B, T)
        Returns:
            encoded features, (B, T, D)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
