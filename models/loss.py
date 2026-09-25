"""
Joint loss for the MAKT model:

    L_total = L_bce + lambda_align * L_align + lambda_mono * L_mono

    L_bce   : main knowledge-tracing binary cross-entropy on valid positions
    L_align : alignment loss — pulls the student cognition snapshot towards
              its most-attended concept embedding and pushes it away from a
              randomly sampled negative concept (margin = delta)
    L_mono  : monotonicity loss — a correct response should not decrease the
              macro ability, an incorrect one should not increase it

The alignment and monotonicity losses accept the per-step histories either
as a Python list of [batch_size, ...] tensors (sequential variant) or as
batched [batch_size, seq_len, ...] tensors (batched-time variant).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAKTLoss(nn.Module):
    """Main BCE + alignment + monotonicity joint loss."""

    def __init__(
        self,
        lambda_align: float = 0.1,
        lambda_mono: float = 0.05,
        delta: float = 0.5,
    ) -> None:
        super().__init__()
        self.lambda_align = lambda_align
        self.lambda_mono = lambda_mono
        self.delta = delta

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        student_states: torch.Tensor | List[torch.Tensor],
        attn_history: torch.Tensor | List[torch.Tensor],
        concept_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            preds           : predicted probabilities, [B, T]
            targets         : ground-truth correctness, [B, T]
            mask            : valid-position mask, [B, T]
            student_states  : per-step cognition snapshots
            attn_history    : per-step concept attention distributions
            concept_matrix  : concept embeddings, [num_concepts, d_model]

        Returns:
            (total_loss, loss_bce, loss_align, loss_mono)
        """
        # Main BCE on valid positions
        loss_bce = F.binary_cross_entropy(preds * mask, targets * mask, reduction="sum") / mask.sum()

        loss_align = self._alignment_loss(student_states, attn_history, concept_matrix)
        loss_mono = self._monotonicity_loss(student_states, targets, mask)

        total_loss = loss_bce + self.lambda_align * loss_align + self.lambda_mono * loss_mono
        return total_loss, loss_bce, loss_align, loss_mono

    # ------------------------------------------------------------------
    # Alignment loss
    # ------------------------------------------------------------------
    def _alignment_loss(
        self,
        student_states: torch.Tensor | List[torch.Tensor],
        attn_history: torch.Tensor | List[torch.Tensor],
        concept_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Pull the student state close to its most relevant concept, push away negatives."""
        if isinstance(student_states, list):
            T = len(student_states)
            if T == 0:
                return torch.tensor(0.0, device=concept_matrix.device)
            total = 0.0
            for t in range(T):
                total += self._alignment_step(student_states[t], attn_history[t], concept_matrix)
            return total / T

        # Batched tensor: [B, T, D] / [B, T, C]
        _, T, _ = student_states.size()
        total = 0.0
        for t in range(T):
            total += self._alignment_step(student_states[:, t, :], attn_history[:, t, :], concept_matrix)
        return total / T

    def _alignment_step(
        self,
        s_t: torch.Tensor,
        a_t: torch.Tensor,
        concept_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Single-step margin loss between the state and positive/negative concepts."""
        batch_size = s_t.size(0)

        pos_idx = torch.argmax(a_t, dim=-1)                    # [B]
        pos_c = concept_matrix[pos_idx]                        # [B, d_model]
        neg_idx = torch.randint(0, concept_matrix.size(0), (batch_size,), device=concept_matrix.device)
        neg_c = concept_matrix[neg_idx]

        pos_dist = torch.norm(s_t - pos_c, p=2, dim=-1)
        neg_dist = torch.norm(s_t - neg_c, p=2, dim=-1)

        return torch.clamp(self.delta + pos_dist - neg_dist, min=0.0).mean()

    # ------------------------------------------------------------------
    # Monotonicity loss
    # ------------------------------------------------------------------
    def _monotonicity_loss(
        self,
        student_states: torch.Tensor | List[torch.Tensor],
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cognition-state direction constraint:
            - a correct response should not decrease the macro ability
            - an incorrect response should not increase the macro ability
        """
        if isinstance(student_states, list):
            T = len(student_states)
            if T <= 1:
                return torch.tensor(0.0, device=targets.device)
            total = 0.0
            valid_pairs = 0
            for t in range(1, T):
                s_prev = student_states[t - 1].mean(dim=-1)    # [B]
                s_curr = student_states[t].mean(dim=-1)        # [B]
                step_loss, step_valid = self._monotonicity_step(
                    s_prev, s_curr, targets[:, t], mask[:, t] * mask[:, t - 1]
                )
                total += step_loss
                valid_pairs += step_valid
            return total / max(valid_pairs, 1.0)

        # Batched tensor: [B, T, D]
        _, T, _ = student_states.size()
        if T <= 1:
            return torch.tensor(0.0, device=targets.device)
        total = 0.0
        valid_pairs = 0
        for t in range(1, T):
            s_prev = student_states[:, t - 1, :].mean(dim=-1)  # [B]
            s_curr = student_states[:, t, :].mean(dim=-1)      # [B]
            step_loss, step_valid = self._monotonicity_step(
                s_prev, s_curr, targets[:, t], mask[:, t] * mask[:, t - 1]
            )
            total += step_loss
            valid_pairs += step_valid
        return total / max(valid_pairs, 1.0)

    @staticmethod
    def _monotonicity_step(
        s_prev: torch.Tensor,
        s_curr: torch.Tensor,
        r_t: torch.Tensor,
        m_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Single-step monotonicity violation, summed over valid adjacent pairs."""
        direction = 2.0 * r_t - 1.0                            # +1 correct, -1 wrong
        violation = torch.clamp(direction * (s_prev - s_curr), min=0.0)
        return (violation * m_t).sum(), m_t.sum().item()
