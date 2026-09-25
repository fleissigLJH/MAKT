"""
Configuration centre for the MAKT project.

All hyper-parameters and file locations are defined here as dataclasses,
grouped by concern:

    DataConfig  : dataset selection, file locations and DataLoader settings
    ModelConfig : MAKT architecture hyper-parameters
    TrainConfig : optimiser, schedule, loss weights and early-stopping settings

The defaults follow the experimental setup reported in the paper:
sequence length 200, 80% of the learners for training and 20% for
validation and testing, learners with fewer than 10 interactions
discarded, d_model = 128, d_state = 32, 2 causal-encoder layers with
8 heads and d_ff = 256, dropout 0.1, batch size 64, Adam with
learning rate 1e-3, StepLR decay 0.5 every 10 epochs, gradient
clipping 1.0, lambda_align = 0.1, lambda_mono = 0.05, margin
delta = 0.5, early stopping on validation AUC with patience 5 after a
minimum of 10 epochs.

NOTE: the string fields marked as placeholders must be replaced with the
real storage paths of the corresponding data files before running.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

# Datasets used in the paper
DATASETS = ("assist2017", "algebra0506", "algebra0607", "moocradar", "xes3g5m")


@dataclass
class DataConfig:
    """Dataset selection, file locations and DataLoader settings."""

    # Which dataset to use: "assist2017", "algebra0506", "algebra0607",
    # "moocradar" or "xes3g5m"
    dataset: str = "assist2017"

    # --- ASSISTments2017 (2017 ASSISTments data mining competition) ---
    # Preprocessed CSV. Expected columns:
    #   user_id, problem_id, skill_id, is_correct, timestamp [, skill_text]
    assist2017_csv: str = "<PATH_TO_ASSISTMENTS2017_CSV>"

    # --- Algebra2005-2006 / Algebra2006-2007 (KDD Cup 2010) ---
    # Preprocessed CSVs with the same schema as ASSISTments2017:
    #   user_id, problem_id, skill_id, is_correct, timestamp [, skill_text]
    algebra0506_csv: str = "<PATH_TO_ALGEBRA2005_2006_CSV>"
    algebra0607_csv: str = "<PATH_TO_ALGEBRA2006_2007_CSV>"

    # --- MoocRadar (coarse-grained mapping version) ---
    # Expected columns: user_id, problem_id, is_correct, skill_id
    mooc_csv: str = "<PATH_TO_MOOCRADAR_INTERACTION_CSV>"
    mooc_nrows: int = 800000        # rows read from the CSV (None = whole file)

    # --- XES3G5M (question-level) ---
    # Expected columns: uid, questions, concepts, responses, timestamps, image2_id
    xes3g5m_csv: str = "<PATH_TO_XES3G5M_QUESTION_LEVEL_CSV>"
    xes3g5m_nrows: int = 444000     # rows read from the CSV (None = whole file)

    # --- Sequence construction (paper protocol) ---
    seq_len: int = 200              # maximum interaction sequence length;
                                    # longer histories are split into new subsequences
    max_text_len: int = 20          # maximum skill-text tokens per interaction
    min_records: int = 10           # learners with fewer interactions are discarded

    # --- Splitting (paper: 80% training, 20% for validation and testing) ---
    train_ratio: float = 0.8
    val_ratio: float = 0.1          # the remaining 0.1 is used for testing
    seed: int = 2024

    # Directory where the built vocabularies are cached
    cache_dir: str = "data_cache"

    # --- DataLoader settings ---
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class ModelConfig:
    """MAKT architecture hyper-parameters (paper defaults)."""

    d_model: int = 128              # model dimension (d)
    d_state: int = 32               # SSM state dimension, sequential variant only
    text_dim: int = 128             # skill-text embedding dimension (Agent_E)
    concept_text_dim: int = 128     # concept base-representation dimension (Agent_C)
    dropout: float = 0.1

    num_concepts: int | None = None  # None = infer from the data

    # Agent_S causal-Transformer encoder (batched-time variant)
    n_heads: int = 8
    d_ff: int = 256
    n_layers: int = 2

    # Sparse top-k concept attention (k); None = dense all-concept attention
    num_attention_concepts: int | None = 128


@dataclass
class TrainConfig:
    """Optimiser, schedule, loss weights and early-stopping settings."""

    epochs: int = 100
    learning_rate: float = 1e-3     # Adam learning rate
    lr_decay_step: int = 10         # StepLR step size
    lr_decay_gamma: float = 0.5     # StepLR decay factor
    clip_grad: float = 1.0          # gradient-norm clipping

    # Early stopping monitors the validation AUC
    early_stop_patience: int = 5    # epochs without improvement before stopping
    min_epochs: int = 10            # never early-stop before this many epochs

    # Joint-loss weights: L = L_bce + lambda_align * L_align + lambda_mono * L_mono
    lambda_align: float = 0.1
    lambda_mono: float = 0.05
    delta: float = 0.5              # margin of the alignment loss

    device: str = "auto"            # "auto" / "cuda" / "cpu"
    log_interval: int = 50
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


@dataclass
class ProjectConfig:
    """Top-level configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def get_config() -> ProjectConfig:
    """Return the default project configuration."""
    return ProjectConfig()


def get_dataset_config(
    dataset: str = "assist2017",
    d_model: int = 128,
    exp_dir: str | None = None,
) -> ProjectConfig:
    """
    Return the configuration for one of the paper datasets.

    Args:
        dataset : one of "assist2017", "algebra0506", "algebra0607",
                  "moocradar", "xes3g5m"
        d_model : model dimension override
        exp_dir : optional experiment directory; checkpoints and logs are
                  placed under <exp_dir>/checkpoints and <exp_dir>/logs
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {DATASETS}.")
    cfg = get_config()
    cfg.data = replace(cfg.data, dataset=dataset)
    cfg.model = replace(cfg.model, d_model=d_model)
    if exp_dir is not None:
        exp_path = Path(exp_dir)
        cfg.train = replace(
            cfg.train,
            checkpoint_dir=str(exp_path / "checkpoints"),
            log_dir=str(exp_path / "logs"),
        )
    return cfg


if __name__ == "__main__":
    cfg = get_config()
    print("Configuration:")
    print(f"  dataset: {cfg.data.dataset}")
    print(f"  seq_len: {cfg.data.seq_len}")
    print(f"  min_records: {cfg.data.min_records}")
    print(f"  d_model: {cfg.model.d_model}")
    print(f"  d_state: {cfg.model.d_state}")
    print(f"  top-k concepts: {cfg.model.num_attention_concepts}")
    print(f"  epochs: {cfg.train.epochs}")
    print(f"  lr: {cfg.train.learning_rate}")
