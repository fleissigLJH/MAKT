# MAKT — Codebase Documentation

Official implementation of the paper:

> **MAKT: Agent-Oriented Cognitive Knowledge Tracing with Concept-Aware State Evolution**

## 1. Overview

**MAKT** (Agent-Oriented Cognitive Knowledge Tracing) decomposes the
classical centralised KT sequence model into **four role-specialized
functional agents**:

| Agent | Module | Cognitive role |
|-------|--------|----------------|
| Agent_E | `ExerciseRepresentationAgent` | Stimulus interpretation: fuses the exercise id, skill text and optional skill id into an exercise embedding |
| Agent_C | `ConceptAssociationAgent` | Concept organization: maintains the global concept matrix and produces the concept attention distribution for the current exercise |
| Agent_S | `CausalTransformerEncoder` / `StudentCognitionAgentMamba` | Cognitive state evolution: evolves the student cognition state over time |
| Agent_F | `FeedbackEvaluationAgent` | Feedback evaluation: reads the cognition state and predicts the probability of a correct response |

The task is classic **knowledge tracing**: given a student's interaction
history, predict the probability that the student answers the next question
correctly.

Note that the agents in MAKT are not autonomous entities in the sense of
conventional multi-agent systems: they are embedded within the
computational architecture of a single model, and all inter-agent
coordination is realized through differentiable latent message passing
under joint optimization.

Two assembly variants of the pipeline are provided:

- **`MAKTSystem`** (default, used by `train.py`) — batched-time variant:
  Agent_E / Agent_C / Agent_F run in parallel over all time steps and
  Agent_S is a causal Transformer encoder over the fused per-step agent
  messages. This is the configuration reported in the paper.
- **`MAKTMambaSystem`** — sequential variant: the system steps through the
  sequence one interaction at a time and Agent_S evolves the concept memory
  matrix `M_t` and the continuous state register `h_t` with a selective
  state-space (Mamba-style) update.

Training uses a joint loss:

```
L_total = L_bce + lambda_align * L_align + lambda_mono * L_mono
```

- `L_bce` — main KT binary cross-entropy on valid positions;
- `L_align` — pulls the student cognition snapshot towards its most-attended
  concept embedding and away from a random negative concept (margin `delta`);
- `L_mono` — monotonicity constraint: a correct response should not decrease
  the macro ability, an incorrect one should not increase it.

---

## 2. Project Structure

```
v2/
├── README.md             <- this document
├── requirements.txt      <- Python dependencies
├── config.py             <- configuration centre (DataConfig / ModelConfig / TrainConfig)
├── train.py              <- training entry point (pure PyTorch loop, no Lightning)
└── makt/                 <- core model package
    ├── __init__.py       <- package exports
    ├── agents.py         <- the four role agents (Agent_E / Agent_C / Agent_S / Agent_F)
    ├── encoder.py        <- causal Transformer encoder (batched-time Agent_S)
    ├── model.py          <- MAKTSystem (batched-time) and MAKTMambaSystem (sequential)
    ├── loss.py           <- joint loss: BCE + alignment + monotonicity
    ├── data.py           <- vocabularies, dataset and loaders for the five benchmarks
    └── metrics.py        <- evaluation metrics: AUC / ACC / F1 / RMSE / MAE
```

---

## 3. File-by-File Description

### Root directory

| File | Purpose |
|------|---------|
| `config.py` | Configuration centre. `DataConfig`: dataset selection, file locations and DataLoader settings; `ModelConfig`: architecture hyper-parameters (d_model, d_state, Agent_S encoder width/depth, sparse top-k concept attention); `TrainConfig`: learning rate, LR decay, joint-loss weights, gradient clipping, early stopping, checkpoint/log directories. Defaults follow the experimental setup in the paper |
| `train.py` | Training entry point. `create_dataloaders()` (dataset-specific preparation), `run_epoch()` (explicit single-epoch train/evaluate implementation), `save_checkpoint()`, and `main()` (full pipeline: data → model → per-epoch training and validation → early stopping → best-checkpoint saving → test evaluation). Supports resuming from an existing checkpoint |

### `makt/` package

| File | Purpose |
|------|---------|
| `agents.py` | The four role agents: `ExerciseRepresentationAgent` (gated fusion of exercise id / skill text / skill id), `ConceptAssociationAgent` (global concept matrix + exercise-to-concept attention), `StudentCognitionAgentMamba` (selective state-space cognition update with concept memory matrix), `FeedbackEvaluationAgent` (prediction MLP) |
| `encoder.py` | `CausalTransformerLayer` / `CausalTransformerEncoder`: pre-norm causal Transformer used as the sequence backbone of Agent_S in the batched-time variant |
| `model.py` | System assembly: `MAKTSystem` (batched-time, default) and `MAKTMambaSystem` (sequential selective-SSM variant). Both share the same four agents |
| `loss.py` | `MAKTLoss`: main BCE + alignment loss (margin over concept distances) + monotonicity loss (cognition-direction constraint). Accepts per-step histories as lists or batched tensors |
| `data.py` | `Vocabulary` (pad/unk symbol vocabulary with pickle caching), `MAKTDataset` (fixed-length sequence construction), `collate_fn`, and the dataset preparations: `prepare_csv_data` (ASSISTments2017 / Algebra2005-2006 / Algebra2006-2007), `prepare_mooc_data` (MoocRadar), `prepare_xes3g5m_data` (XES3G5M) |
| `metrics.py` | `compute_accuracy()` and `evaluate_all()` — AUC / ACC / F1 / RMSE / MAE on the valid (masked) positions |
| `__init__.py` | Package exports — the unified public interface |

---

## 4. Dataset Sources

The five benchmarks used in the paper can be obtained from the following
official sources:

| Dataset | Source |
|---------|--------|
| ASSISTments2017 | https://sites.google.com/view/assistmentsdatamining/data-mining-competition-2017 |
| Algebra2005-2006 | https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp |
| Algebra2006-2007 | https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp |
| MoocRadar | https://github.com/THU-KEG/MOOC-Radar |
| XES3G5M | https://github.com/ai4ed/XES3G5M |

### Preprocessing protocol (as in the paper)

1. Each question-answering event is treated as one interaction.
2. Learners whose interaction sequences contain fewer than 10 records are
   discarded (`DataConfig.min_records`).
3. The remaining records of each learner are sorted chronologically.
4. Sequences are truncated to at most 200 time steps
   (`DataConfig.seq_len`); the exceeding portions are split into new
   subsequences.
5. 80% of the learners are used for training and 20% for validation and
   testing (10% / 10%, `DataConfig.train_ratio` / `val_ratio`).

---

## 5. External Data Files (paths must be filled in before running)

All file locations are marked with **human-readable placeholder strings** in
`config.py` (`DataConfig`). Replace them with the real paths before running:

| Configuration field | Description |
|---------------------|-------------|
| `DataConfig.assist2017_csv` | `<PATH_TO_ASSISTMENTS2017_CSV>` — preprocessed ASSISTments2017 CSV. Expected columns: `user_id, problem_id, skill_id, is_correct, timestamp` (optional: `skill_text`) |
| `DataConfig.algebra0506_csv` | `<PATH_TO_ALGEBRA2005_2006_CSV>` — preprocessed Algebra2005-2006 CSV with the same schema as above |
| `DataConfig.algebra0607_csv` | `<PATH_TO_ALGEBRA2006_2007_CSV>` — preprocessed Algebra2006-2007 CSV with the same schema as above |
| `DataConfig.mooc_csv` | `<PATH_TO_MOOCRADAR_INTERACTION_CSV>` — pre-processed MoocRadar interaction CSV (coarse-grained mapping version). Expected columns: `user_id, problem_id, is_correct, skill_id` |
| `DataConfig.xes3g5m_csv` | `<PATH_TO_XES3G5M_QUESTION_LEVEL_CSV>` — XES3G5M question-level interaction CSV. Expected columns: `uid, questions, concepts, responses, timestamps, image2_id` |

Notes:

- `DataConfig.dataset` selects the dataset: `"assist2017"`,
  `"algebra0506"`, `"algebra0607"`, `"moocradar"` or `"xes3g5m"` (can also
  be set via `python train.py --dataset ...`).
- For ASSISTments2017 and the two Algebra datasets, a shared preprocessed
  CSV schema is expected: `user_id, problem_id, skill_id, is_correct,
  timestamp`. If a `skill_text` column with the skill name is present it is
  used as the text input of Agent_E; otherwise the skill id is used.
- `mooc_nrows` / `xes3g5m_nrows` limit the number of CSV rows read
  (set to `None` to load the whole file).
- Built vocabularies are cached under `DataConfig.cache_dir`; delete the
  cache when the data or preprocessing changes.

---

## 6. Environment and Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

The main dependencies are `torch`, `numpy`, `pandas` and `scikit-learn`.

### Steps to run

1. Prepare the data file of the chosen dataset and fill in its real path in
   `config.py` (`DataConfig`).
2. Optionally adjust `ModelConfig` (d_model, Agent_S encoder settings,
   sparse top-k concept attention) and `TrainConfig` (learning rate, epochs,
   loss weights, early-stopping parameters).
3. Start training:

```bash
# ASSISTments2017 (default)
python train.py --dataset assist2017

# Algebra2005-2006 / Algebra2006-2007
python train.py --dataset algebra0506 --exp_dir runs/algebra0506
python train.py --dataset algebra0607 --exp_dir runs/algebra0607

# MoocRadar / XES3G5M, with an experiment directory for checkpoints and logs
python train.py --dataset moocradar --exp_dir runs/moocradar
python train.py --dataset xes3g5m  --exp_dir runs/xes3g5m

# Common overrides
python train.py --dataset assist2017 --d_model 256 \
    --lambda_align 0.1 --lambda_mono 0.05 --delta 0.5
```

### Default configuration (as reported in the paper)

| Hyper-parameter | Value |
|-----------------|-------|
| Sequence length | 200 |
| d_model / d_state | 128 / 32 |
| Agent_S encoder | 2 layers, 8 heads, d_ff = 256 |
| Dropout | 0.1 |
| Sparse top-k concept attention | k = 128 |
| Batch size | 64 |
| Optimiser | Adam, learning rate 1e-3 |
| LR schedule | StepLR, ×0.5 every 10 epochs |
| Gradient clipping | max norm 1.0 |
| Loss weights | lambda_align = 0.1, lambda_mono = 0.05, delta = 0.5 |
| Early stopping | validation AUC, patience 5, min 10 epochs |

### Training behaviour

- Every epoch prints the joint loss (and its BCE / alignment / monotonicity
  parts) plus AUC, ACC, F1, RMSE and MAE for both the training and
  validation sets. (The paper reports AUC, ACC, RMSE and MAE; F1 is
  provided here for convenience.)
- Early stopping monitors the **validation AUC**: it never triggers before
  `min_epochs` (default 10) epochs, and stops after `early_stop_patience`
  (default 5) consecutive epochs without improvement.
- Whenever the validation AUC improves, the checkpoint (model + optimiser +
  epoch + metrics) is saved to `<checkpoint_dir>/best_model.pt`. If this
  file already exists when training starts, training **resumes** from it.
- After training, the per-epoch history is dumped to
  `<log_dir>/training_history.json`, the best checkpoint is reloaded and
  evaluated on the test set, and the result is written to
  `<log_dir>/test_metrics.json`.
- The sequential `MAKTMambaSystem` variant can be used by importing it from
  the `makt` package and passing it to the same training loop (its per-step
  list outputs are accepted by `MAKTLoss` as well).
