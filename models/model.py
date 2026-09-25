"""
MAKT system assembly.

Two assembly variants of the four-agent pipeline
(Agent_E -> Agent_C -> Agent_S -> Agent_F) are provided:

    MAKTSystem       : batched-time variant (default, used by train.py).
                       Agent_E / Agent_C / Agent_F run in parallel over all
                       time steps; Agent_S is a causal Transformer encoder
                       over the fused per-step agent messages.
    MAKTMambaSystem  : sequential variant. The system steps through the
                       sequence one interaction at a time and Agent_S evolves
                       the cognition state with the selective state-space
                       (Mamba-style) update defined in agents.py.

Both variants share the same four agents and the same joint loss
(makt.loss.MAKTLoss): main BCE + alignment + monotonicity.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .agents import (
    ConceptAssociationAgent,
    ExerciseRepresentationAgent,
    FeedbackEvaluationAgent,
    StudentCognitionAgentMamba,
)
from .encoder import CausalTransformerEncoder


# ---------------------------------------------------------------------------
# Batched-time variant (default)
# ---------------------------------------------------------------------------
class MAKTSystem(nn.Module):
    """
    Batched-time Multi-Agent Knowledge Tracing system.

    Inputs:
        problem_ids  : [batch_size, seq_len]
        skill_texts  : [batch_size, seq_len, text_len]
        prev_correct : [batch_size, seq_len]
        skill_ids    : [batch_size, seq_len] (optional)

    Outputs:
        predictions             : [batch_size, seq_len]
        attn_weights_history    : [batch_size, seq_len, num_concepts]
        student_states_history  : [batch_size, seq_len, d_model]
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        num_concepts: int,
        problem_vocab_size: int,
        skill_text_vocab_size: int,
        skill_id_vocab_size: int | None = None,
        text_dim: int = 128,
        concept_text_dim: int = 128,
        dropout: float = 0.1,
        n_heads: int = 8,
        d_ff: int = 256,
        n_layers: int = 2,
        num_attention_concepts: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_concepts = num_concepts
        # Cap the number of attended concepts for efficiency on
        # large-concept datasets.
        self.num_attention_concepts = num_attention_concepts or num_concepts
        # Fall back to the global concept matrix when the concept count is
        # large (avoids the [B, C, D] memory blowup).
        self.sparse_concepts = num_concepts > 512

        # Agent_E: per-step exercise representation
        self.agent_E = ExerciseRepresentationAgent(
            d_model=d_model,
            problem_vocab_size=problem_vocab_size,
            skill_text_vocab_size=skill_text_vocab_size,
            skill_id_vocab_size=skill_id_vocab_size,
            text_dim=text_dim,
            dropout=dropout,
        )

        # Agent_C: concept association for each exercise embedding
        self.agent_C = ConceptAssociationAgent(
            d_model=d_model,
            num_concepts=num_concepts,
            concept_text_dim=concept_text_dim,
            dropout=dropout,
        )

        # Agent_S: causal Transformer encoder over the fused messages
        # Input message: [exercise ; concept ; memory read ; previous response]
        self.fusion_proj = nn.Linear(d_model * 3 + 1, d_model)
        self.agent_S = CausalTransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Concept memory update, weighted by the attention history
        self.memory_gate = nn.Linear(d_model, d_model)

        # Agent_F: prediction from the cognition snapshot and exercise embedding
        self.agent_F = FeedbackEvaluationAgent(d_model=d_model, dropout=dropout)

    def forward(
        self,
        problem_ids: torch.Tensor,
        skill_texts: torch.Tensor,
        prev_correct: torch.Tensor,
        skill_ids: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = problem_ids.size()

        # ---- Agent_E: exercise representation (parallel across time) ----
        ex_ids_flat = problem_ids.reshape(-1)                          # [B*T]
        ex_texts_flat = skill_texts.reshape(batch_size * seq_len, -1)  # [B*T, text_len]
        skill_ids_flat = skill_ids.reshape(-1) if skill_ids is not None else None

        e_flat = self.agent_E(ex_ids_flat, ex_texts_flat, skill_ids_flat)  # [B*T, D]
        e_t = e_flat.view(batch_size, seq_len, self.d_model)               # [B, T, D]

        # ---- Agent_C: concept association (parallel across time) ----
        c_flat, a_flat_full = self.agent_C(e_flat)  # [B*T, D], [B*T, num_concepts]
        c_t = c_flat.view(batch_size, seq_len, self.d_model)

        # Optional top-k concept attention for large-concept datasets
        if self.num_attention_concepts < self.num_concepts:
            topk_vals, topk_idx = torch.topk(a_flat_full, self.num_attention_concepts, dim=-1)
            topk_vals = F.softmax(topk_vals, dim=-1)
            a_flat = torch.zeros_like(a_flat_full)
            a_flat.scatter_(1, topk_idx, topk_vals)
        else:
            a_flat = a_flat_full
        a_t = a_flat.view(batch_size, seq_len, self.num_concepts)      # [B, T, C]

        # ---- Agent_S: causal cognition evolution ----
        prev_correct_exp = prev_correct.unsqueeze(-1)                  # [B, T, 1]

        # Memory read at each step, weighted by the concept attention
        C = self.agent_C.get_concept_matrix()                          # [C, D]
        M_read = torch.bmm(a_t, C.unsqueeze(0).expand(batch_size, -1, -1))  # [B, T, D]

        fused = torch.cat([e_t, c_t, M_read, prev_correct_exp], dim=-1)    # [B, T, 3D+1]
        x = F.silu(self.fusion_proj(fused))                                # [B, T, D]

        valid_mask = (problem_ids != 0).float()                        # [B, T]
        s_t = self.agent_S(x, mask=valid_mask)                         # [B, T, D]

        # Concept memory update (keeps the M_t semantics per step)
        if self.sparse_concepts:
            # For prediction, Agent_F reads directly from the global
            # concept matrix C with the attention weights a_t.
            M_flat = C.unsqueeze(0).expand(batch_size * seq_len, -1, -1)
        else:
            delta_M = torch.bmm(a_t.transpose(1, 2), s_t)              # [B, C, D]
            gate = torch.sigmoid(self.memory_gate(delta_M))
            M_t = (1.0 - gate) * C.unsqueeze(0) + gate * torch.tanh(delta_M)
            # Replicate the memory matrix over time and flatten
            M_t_expanded = M_t.unsqueeze(2).expand(-1, -1, seq_len, -1)    # [B, C, T, D]
            M_flat = M_t_expanded.permute(0, 2, 1, 3).reshape(
                batch_size * seq_len, self.num_concepts, self.d_model
            )

        # ---- Agent_F: prediction ----
        s_flat = s_t.reshape(batch_size * seq_len, self.d_model)
        e_flat_for_f = e_t.reshape(batch_size * seq_len, self.d_model)
        a_flat_for_f = a_t.reshape(batch_size * seq_len, self.num_concepts)
        y_flat, _ = self.agent_F(M_flat, e_flat_for_f, a_flat_for_f)   # [B*T, 1]
        predictions = y_flat.view(batch_size, seq_len)

        return predictions, a_t, s_t


# Backward-compatible alias (the batched-time system was formerly "fast").
MAKTFastSystem = MAKTSystem


# ---------------------------------------------------------------------------
# Sequential variant (selective state-space Agent_S)
# ---------------------------------------------------------------------------
class MAKTMambaSystem(nn.Module):
    """
    Sequential Multi-Agent Knowledge Tracing system.

    Steps through the interaction sequence one time step at a time; Agent_S
    evolves the concept memory matrix M_t and the continuous state register
    h_t with a selective state-space update.

    Inputs:
        problem_ids  : [batch_size, seq_len]
        skill_texts  : [batch_size, seq_len, text_len]
        prev_correct : [batch_size, seq_len]
        skill_ids    : [batch_size, seq_len] (optional)

    Outputs:
        predictions             : [batch_size, seq_len]
        attn_weights_history    : List[[batch_size, num_concepts]] (per step)
        student_states_history  : List[[batch_size, d_model]] (per step)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        num_concepts: int,
        problem_vocab_size: int,
        skill_text_vocab_size: int,
        skill_id_vocab_size: int | None = None,
        text_dim: int = 128,
        concept_text_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_concepts = num_concepts

        self.agent_E = ExerciseRepresentationAgent(
            d_model=d_model,
            problem_vocab_size=problem_vocab_size,
            skill_text_vocab_size=skill_text_vocab_size,
            skill_id_vocab_size=skill_id_vocab_size,
            text_dim=text_dim,
            dropout=dropout,
        )
        self.agent_C = ConceptAssociationAgent(
            d_model=d_model,
            num_concepts=num_concepts,
            concept_text_dim=concept_text_dim,
            dropout=dropout,
        )
        self.agent_S = StudentCognitionAgentMamba(
            d_model=d_model,
            d_state=d_state,
            num_concepts=num_concepts,
            dropout=dropout,
        )
        self.agent_F = FeedbackEvaluationAgent(d_model=d_model, dropout=dropout)

    def forward(
        self,
        problem_ids: torch.Tensor,
        skill_texts: torch.Tensor,
        prev_correct: torch.Tensor,
        skill_ids: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        batch_size, seq_len = problem_ids.size()
        device = problem_ids.device

        # Initial cognition states
        M_t = torch.zeros(batch_size, self.num_concepts, self.d_model, device=device)
        h_t = torch.zeros(batch_size, self.d_model, self.d_state, device=device)

        outputs: List[torch.Tensor] = []
        attn_weights_history: List[torch.Tensor] = []
        student_states_history: List[torch.Tensor] = []

        # Step-by-step agent collaboration
        for t in range(seq_len):
            ex_ids_t = problem_ids[:, t]
            ex_texts_t = skill_texts[:, t, :]
            r_prev_t = prev_correct[:, t].unsqueeze(-1)                # [B, 1]

            # Agent_E: exercise representation
            e_t = self.agent_E(
                ex_ids_t,
                ex_texts_t,
                skill_ids[:, t] if skill_ids is not None else None,
            )

            # Agent_C: concept association
            c_t, a_t = self.agent_C(e_t)

            # Agent_S: cognition state evolution
            M_t, h_t = self.agent_S(e_t, c_t, r_prev_t, M_t, h_t, a_t)

            # Agent_F: prediction and feedback
            y_t, s_t = self.agent_F(M_t, e_t, a_t)

            outputs.append(y_t)
            attn_weights_history.append(a_t)
            student_states_history.append(s_t)

        predictions = torch.cat(outputs, dim=-1)                       # [B, seq_len]
        return predictions, attn_weights_history, student_states_history
