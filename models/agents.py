"""
Core agents of MAKT (Agent-Oriented Cognitive Knowledge Tracing with
Concept-Aware State Evolution).

The classical centralised KT sequence model is decomposed into four
role-specialized functional agents:

    1. ExerciseRepresentationAgent  (Agent_E, stimulus interpretation):
       fuses the exercise id, the skill text and the optional skill id into
       an exercise embedding e_t.
    2. ConceptAssociationAgent      (Agent_C, concept organization):
       maintains the global concept matrix and computes the concept
       attention distribution a_t and the concept-aware embedding c_t for
       the current exercise.
    3. StudentCognitionAgentMamba   (Agent_S, cognitive state evolution):
       evolves the student cognition state (concept memory matrix M_t and
       continuous state register h_t) with a selective state-space
       (Mamba-style) update.
    4. FeedbackEvaluationAgent      (Agent_F, feedback evaluation):
       reads the cognition state and produces the predicted correctness
       probability y_t.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Exercise Representation Agent (Agent_E)
# ---------------------------------------------------------------------------
class ExerciseRepresentationAgent(nn.Module):
    """
    Fuses the exercise id and the skill text into an exercise embedding e_t.

    Inputs:
        problem_ids : [batch_size] or [batch_size, 1]
        skill_texts : [batch_size, text_len] skill-text token ids
        skill_ids   : [batch_size] or [batch_size, 1] (optional)

    Output:
        e_t : [batch_size, d_model]
    """

    def __init__(
        self,
        d_model: int,
        problem_vocab_size: int,
        skill_text_vocab_size: int,
        skill_id_vocab_size: int | None = None,
        text_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Exercise id embedding
        self.problem_embedding = nn.Embedding(problem_vocab_size, d_model, padding_idx=0)

        # Skill-text encoding: embedding lookup -> masked mean pooling -> projection
        self.text_encoder = nn.Embedding(skill_text_vocab_size, text_dim, padding_idx=0)
        self.text_projector = nn.Linear(text_dim, d_model)

        # Optional skill id embedding
        self.use_skill_id = skill_id_vocab_size is not None
        if self.use_skill_id:
            self.skill_id_embedding = nn.Embedding(skill_id_vocab_size, d_model, padding_idx=0)

        # Gated fusion
        fusion_dim = d_model * 3 if self.use_skill_id else d_model * 2
        self.gate_linear = nn.Linear(fusion_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        problem_ids: torch.Tensor,
        skill_texts: torch.Tensor,
        skill_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if problem_ids.dim() == 2:
            problem_ids = problem_ids.squeeze(-1)
        if skill_ids is not None and skill_ids.dim() == 2:
            skill_ids = skill_ids.squeeze(-1)

        # Id feature
        h_problem = self.problem_embedding(problem_ids)  # [B, d_model]

        # Text feature: mean pooling over non-padded tokens
        text_embeds = self.text_encoder(skill_texts)          # [B, text_len, text_dim]
        mask = (skill_texts != 0).unsqueeze(-1).float()       # [B, text_len, 1]
        text_sum = (text_embeds * mask).sum(dim=1)            # [B, text_dim]
        text_len = mask.sum(dim=1).clamp(min=1.0)             # [B, 1]
        text_pooled = text_sum / text_len                     # [B, text_dim]
        h_text = F.silu(self.text_projector(text_pooled))     # [B, d_model]

        # Optional skill id feature
        parts = [h_problem, h_text]
        if self.use_skill_id and skill_ids is not None:
            h_skill_id = self.skill_id_embedding(skill_ids)
            parts.append(h_skill_id)

        concat_feats = torch.cat(parts, dim=-1)               # [B, fusion_dim]
        gate = torch.sigmoid(self.gate_linear(concat_feats))  # [B, d_model]

        # Gated residual fusion
        e_t = gate * h_text + (1.0 - gate) * h_problem
        if self.use_skill_id and skill_ids is not None:
            e_t = e_t + 0.1 * h_skill_id

        return self.layer_norm(self.dropout(e_t))


# ---------------------------------------------------------------------------
# 2. Concept Association Agent (Agent_C)
# ---------------------------------------------------------------------------
class ConceptAssociationAgent(nn.Module):
    """
    Maintains the global concept embeddings and computes attention-based
    concept association weights for the current exercise.

    Input:
        e_t : [batch_size, d_model]

    Outputs:
        c_t : [batch_size, d_model] aggregated concept embedding
        a_t : [batch_size, num_concepts] concept attention distribution
    """

    def __init__(
        self,
        d_model: int,
        num_concepts: int,
        concept_text_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_concepts = num_concepts

        # Learnable concept base representations (may be initialised from
        # pre-trained concept text vectors)
        self.concept_text_embeds = nn.Parameter(torch.randn(num_concepts, concept_text_dim))
        nn.init.xavier_uniform_(self.concept_text_embeds)
        self.concept_projector = nn.Linear(concept_text_dim, d_model)

        # Exercise -> concept attention projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.scale = math.sqrt(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def get_concept_matrix(self) -> torch.Tensor:
        """Return all concept embeddings in d_model space, [num_concepts, d_model]."""
        return F.silu(self.concept_projector(self.concept_text_embeds))

    def forward(self, e_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = e_t.size(0)

        # Global concept representations
        C = self.get_concept_matrix()                            # [num_concepts, d_model]
        C_expanded = C.unsqueeze(0).expand(batch_size, -1, -1)   # [B, num_concepts, d_model]

        # Attention
        Q = self.w_q(e_t).unsqueeze(1)                           # [B, 1, d_model]
        K = self.w_k(C_expanded)                                 # [B, num_concepts, d_model]
        V = self.w_v(C_expanded)                                 # [B, num_concepts, d_model]

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [B, 1, num_concepts]
        a_t = F.softmax(attn_scores, dim=-1).squeeze(1)             # [B, num_concepts]

        # Weighted aggregation
        c_t = torch.bmm(a_t.unsqueeze(1), V).squeeze(1)          # [B, d_model]

        return self.layer_norm(self.dropout(c_t)), a_t


# ---------------------------------------------------------------------------
# 3. Student Cognition Agent (Agent_S) with a selective state-space update
# ---------------------------------------------------------------------------
class StudentCognitionAgentMamba(nn.Module):
    """
    Maintains the student cognition state with a selective state-space
    (Mamba-style) update.

    State:
        M_t : [batch_size, num_concepts, d_model] concept-level memory matrix
        h_t : [batch_size, d_model, d_state] continuous state register

    Inputs:
        e_t    : [batch_size, d_model] exercise embedding
        c_t    : [batch_size, d_model] concept embedding
        r_prev : [batch_size, 1] previous response
        M_prev : [batch_size, num_concepts, d_model]
        h_prev : [batch_size, d_model, d_state]
        a_t    : [batch_size, num_concepts] concept attention distribution

    Outputs:
        M_t, h_t : updated cognition state
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        num_concepts: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_concepts = num_concepts

        # Cognitive stimulus: [e_t ; c_t ; r_prev]
        self.input_projection = nn.Linear(d_model * 2 + 1, d_model)

        # Selective SSM projections
        self.x_proj = nn.Linear(d_model, d_state * 2 + d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)

        # State matrix A in log space (guarantees negative values = stable decay)
        A_init = torch.arange(1, d_state + 1).float().repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A_init))

        self.out_proj = nn.Linear(d_model, d_model)
        self.memory_gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        e_t: torch.Tensor,
        c_t: torch.Tensor,
        r_prev: torch.Tensor,
        M_prev: torch.Tensor,
        h_prev: torch.Tensor,
        a_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Build the cognitive stimulus
        stimulus = torch.cat([e_t, c_t, r_prev], dim=-1)       # [B, 2*d_model + 1]
        u_t = F.silu(self.input_projection(stimulus))          # [B, d_model]
        u_t = self.dropout(u_t)

        # 2. Selective parameterisation
        A = -torch.exp(self.A_log)                             # [d_model, d_state]
        x_db = self.x_proj(u_t)                                # [B, 2*d_state + d_model]
        dt, B, C = torch.split(x_db, [self.d_model, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                      # [B, d_model]

        # 3. Discretised state transition
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))      # [B, d_model, d_state]
        dB = dt.unsqueeze(-1) * B.unsqueeze(1) * u_t.unsqueeze(-1)  # [B, d_model, d_state]
        h_t = dA * h_prev + dB                                 # [B, d_model, d_state]

        # 4. Read out the cognition signal
        y_mamba = torch.bmm(h_t, C.unsqueeze(-1)).squeeze(-1)  # [B, d_model]
        y_mamba = F.silu(self.out_proj(y_mamba))

        # 5. Update the concept memory matrix (attention-weighted, gated)
        update_mask = a_t.unsqueeze(-1)                        # [B, num_concepts, 1]
        delta_M = update_mask * y_mamba.unsqueeze(1)           # [B, num_concepts, d_model]
        gate = torch.sigmoid(self.memory_gate(delta_M))        # [B, num_concepts, d_model]
        M_t = (1.0 - gate) * M_prev + gate * torch.tanh(delta_M)

        return M_t, h_t


# ---------------------------------------------------------------------------
# 4. Feedback Evaluation Agent (Agent_F)
# ---------------------------------------------------------------------------
class FeedbackEvaluationAgent(nn.Module):
    """
    Produces the predicted correctness probability y_t from the cognition
    state (feedback evaluation).

    Inputs:
        M_t : [batch_size, num_concepts, d_model] concept memory matrix
        e_t : [batch_size, d_model] exercise embedding
        a_t : [batch_size, num_concepts] concept attention distribution

    Outputs:
        y_t : [batch_size, 1] predicted correctness probability
        s_t : [batch_size, d_model] student cognition snapshot
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.prediction_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        M_t: torch.Tensor,
        e_t: torch.Tensor,
        a_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Read the relevant cognition state
        s_t = torch.bmm(a_t.unsqueeze(1), M_t).squeeze(1)      # [B, d_model]

        # Joint prediction from the cognition snapshot and the exercise embedding
        concat_feat = torch.cat([s_t, e_t], dim=-1)            # [B, 2*d_model]
        logits = self.prediction_mlp(concat_feat)              # [B, 1]
        y_t = torch.sigmoid(logits)

        return y_t, s_t


# Backward-compatible alias (Agent_F was formerly "EnvironmentFeedbackAgent").
EnvironmentFeedbackAgent = FeedbackEvaluationAgent
