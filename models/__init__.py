"""
MAKT: Agent-Oriented Cognitive Knowledge Tracing with Concept-Aware
State Evolution
============================================================================

A modular implementation of the MAKT knowledge-tracing model, which
decomposes the classical centralised KT sequence model into four
role-specialized functional agents:

    Agent_E  ExerciseRepresentationAgent   stimulus interpretation
    Agent_C  ConceptAssociationAgent       concept organization
    Agent_S  StudentCognitionAgentMamba / CausalTransformerEncoder
                                           cognitive state evolution
    Agent_F  FeedbackEvaluationAgent       feedback evaluation

Package layout:

    makt/
        agents.py    the four role agents
        encoder.py   causal Transformer encoder (batched-time Agent_S)
        model.py     MAKTSystem (batched-time, default) and
                     MAKTMambaSystem (sequential selective-SSM variant)
        loss.py      joint loss (BCE + alignment + monotonicity)
        data.py      vocabularies, dataset and DataLoader preparation for
                     ASSISTments2017 / Algebra2005-2006 / Algebra2006-2007 /
                     MoocRadar / XES3G5M
        metrics.py   evaluation metrics (AUC / ACC / F1 / RMSE / MAE)
    config.py        all configuration dataclasses
    train.py         pure-PyTorch training entry point
"""

from .agents import (
    ConceptAssociationAgent,
    EnvironmentFeedbackAgent,
    ExerciseRepresentationAgent,
    FeedbackEvaluationAgent,
    StudentCognitionAgentMamba,
)
from .data import (
    MAKTDataset,
    Vocabulary,
    collate_fn,
    filter_short_sequences,
    prepare_csv_data,
    prepare_mooc_data,
    prepare_xes3g5m_data,
)
from .encoder import CausalTransformerEncoder
from .loss import MAKTLoss
from .metrics import compute_accuracy, evaluate_all
from .model import MAKTSystem, MAKTFastSystem, MAKTMambaSystem

__all__ = [
    # Agents
    "ExerciseRepresentationAgent",
    "ConceptAssociationAgent",
    "StudentCognitionAgentMamba",
    "FeedbackEvaluationAgent",
    "EnvironmentFeedbackAgent",
    # Systems
    "MAKTSystem",
    "MAKTFastSystem",
    "MAKTMambaSystem",
    "CausalTransformerEncoder",
    # Loss
    "MAKTLoss",
    # Data
    "Vocabulary",
    "MAKTDataset",
    "collate_fn",
    "filter_short_sequences",
    "prepare_csv_data",
    "prepare_mooc_data",
    "prepare_xes3g5m_data",
    # Metrics
    "compute_accuracy",
    "evaluate_all",
]
