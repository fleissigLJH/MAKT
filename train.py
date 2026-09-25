"""
Pure-PyTorch training entry point for the MAKT model.

Supports the five benchmark datasets used in the paper — ASSISTments2017,
Algebra2005-2006, Algebra2006-2007, MoocRadar and XES3G5M (selected
through DataConfig.dataset or the --dataset flag).

Pipeline:
    1. Build the train / val / test DataLoaders (makt.data).
    2. Instantiate the batched-time MAKTSystem with the configured
       hyper-parameters.
    3. Train with Adam + StepLR, monitoring the validation AUC with early
       stopping (which never triggers before `min_epochs` epochs).
    4. Save the best checkpoint (highest validation AUC), dump the training
       history, then reload the best checkpoint and evaluate on the test set
       (AUC / ACC / F1 / RMSE / MAE).

Run:
    python train.py --dataset assist2017
    python train.py --dataset moocradar --exp_dir runs/mooc
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from config import DATASETS, ProjectConfig, get_config, get_dataset_config
from makt import (
    MAKTLoss,
    MAKTSystem,
    collate_fn,
    evaluate_all,
    prepare_csv_data,
    prepare_mooc_data,
    prepare_xes3g5m_data,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MAKT model.")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=list(DATASETS))
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--exp_dir", type=str, default=None,
                        help="experiment directory for checkpoints/logs")
    parser.add_argument("--nrows", type=int, default=None,
                        help="override the CSV row limit (moocradar / xes3g5m)")
    parser.add_argument("--lambda_align", type=float, default=None,
                        help="alignment loss weight")
    parser.add_argument("--lambda_mono", type=float, default=None,
                        help="monotonicity loss weight")
    parser.add_argument("--delta", type=float, default=None,
                        help="alignment margin delta")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ProjectConfig:
    """Build the project configuration and apply the command-line overrides."""
    if args.dataset is not None or args.d_model is not None:
        config = get_dataset_config(
            dataset=args.dataset or "assist2017",
            d_model=args.d_model or 128,
            exp_dir=args.exp_dir,
        )
    else:
        config = get_config()

    if args.exp_dir is not None:
        exp_path = Path(args.exp_dir)
        config.train = replace(
            config.train,
            checkpoint_dir=str(exp_path / "checkpoints"),
            log_dir=str(exp_path / "logs"),
        )

    if args.nrows is not None:
        if config.data.dataset == "moocradar":
            config.data = replace(config.data, mooc_nrows=args.nrows)
        elif config.data.dataset == "xes3g5m":
            config.data = replace(config.data, xes3g5m_nrows=args.nrows)

    train_overrides = {
        k: v for k, v in {
            "lambda_align": args.lambda_align,
            "lambda_mono": args.lambda_mono,
            "delta": args.delta,
        }.items() if v is not None
    }
    if train_overrides:
        config.train = replace(config.train, **train_overrides)

    return config


def get_device(device_preference: str) -> torch.device:
    if device_preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_preference)


# ---------------------------------------------------------------------------
# Data and model construction
# ---------------------------------------------------------------------------
def create_dataloaders(config: ProjectConfig) -> Dict[str, object]:
    """Prepare the datasets and return the DataLoaders plus vocabulary sizes."""
    csv_fields = {
        "assist2017": config.data.assist2017_csv,
        "algebra0506": config.data.algebra0506_csv,
        "algebra0607": config.data.algebra0607_csv,
    }
    if config.data.dataset in csv_fields:
        cache_dir = Path(config.data.cache_dir) / config.data.dataset
        cache_dir.mkdir(parents=True, exist_ok=True)
        datasets_and_vocabs = prepare_csv_data(
            csv_path=csv_fields[config.data.dataset],
            seq_len=config.data.seq_len,
            max_text_len=config.data.max_text_len,
            min_records=config.data.min_records,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            seed=config.data.seed,
            cache_dir=cache_dir,
        )
    elif config.data.dataset == "moocradar":
        cache_dir = Path(config.data.cache_dir) / "mooc"
        cache_dir.mkdir(parents=True, exist_ok=True)
        datasets_and_vocabs = prepare_mooc_data(
            csv_path=config.data.mooc_csv,
            seq_len=config.data.seq_len,
            max_text_len=config.data.max_text_len,
            min_records=config.data.min_records,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            seed=config.data.seed,
            cache_dir=cache_dir,
            nrows=config.data.mooc_nrows,
        )
    elif config.data.dataset == "xes3g5m":
        cache_dir = Path(config.data.cache_dir) / "xes3g5m"
        cache_dir.mkdir(parents=True, exist_ok=True)
        datasets_and_vocabs = prepare_xes3g5m_data(
            csv_path=config.data.xes3g5m_csv,
            seq_len=config.data.seq_len,
            max_text_len=config.data.max_text_len,
            min_records=config.data.min_records,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            seed=config.data.seed,
            cache_dir=cache_dir,
            nrows=config.data.xes3g5m_nrows,
        )
    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset}")

    train_ds, val_ds, test_ds, p_vocab, s_vocab, st_vocab = datasets_and_vocabs

    loader_kwargs = {
        "batch_size": config.data.batch_size,
        "num_workers": config.data.num_workers,
        "pin_memory": config.data.pin_memory,
        "collate_fn": collate_fn,
    }
    return {
        "train": DataLoader(train_ds, shuffle=True, **loader_kwargs),
        "val": DataLoader(val_ds, shuffle=False, **loader_kwargs),
        "test": DataLoader(test_ds, shuffle=False, **loader_kwargs),
        "problem_vocab_size": len(p_vocab),
        "skill_id_vocab_size": len(s_vocab),
        "skill_text_vocab_size": len(st_vocab),
        "num_concepts": len(s_vocab),
    }


def create_model(config: ProjectConfig, data_info: Dict[str, int]) -> MAKTSystem:
    num_concepts = config.model.num_concepts or data_info["num_concepts"]
    return MAKTSystem(
        d_model=config.model.d_model,
        d_state=config.model.d_state,
        num_concepts=num_concepts,
        problem_vocab_size=data_info["problem_vocab_size"],
        skill_text_vocab_size=data_info["skill_text_vocab_size"],
        skill_id_vocab_size=data_info["skill_id_vocab_size"],
        text_dim=config.model.text_dim,
        concept_text_dim=config.model.concept_text_dim,
        dropout=config.model.dropout,
        n_heads=config.model.n_heads,
        d_ff=config.model.d_ff,
        n_layers=config.model.n_layers,
        num_attention_concepts=config.model.num_attention_concepts,
    )


# ---------------------------------------------------------------------------
# Single-epoch training / evaluation
# ---------------------------------------------------------------------------
def run_epoch(
    model: MAKTSystem,
    criterion: MAKTLoss,
    loader: DataLoader,
    device: torch.device,
    clip_grad: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Run one training or evaluation epoch.

    Args:
        optimizer : if given, run in training mode and update the weights;
                    if None, run in evaluation mode

    Returns:
        (avg_loss, metrics_dict)
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss, total_bce, total_align, total_mono = 0.0, 0.0, 0.0, 0.0
    num_batches = 0

    all_labels: List[torch.Tensor] = []
    all_preds: List[torch.Tensor] = []
    all_masks: List[torch.Tensor] = []

    for batch in loader:
        problem_ids = batch["problem_ids"].to(device)
        skill_ids = batch["skill_ids"].to(device)
        skill_texts = batch["skill_texts"].to(device)
        prev_correct = batch["prev_correct"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)

        with torch.set_grad_enabled(is_training):
            predictions, attns, states = model(
                problem_ids=problem_ids,
                skill_texts=skill_texts,
                prev_correct=prev_correct,
                skill_ids=skill_ids,
            )

            concept_matrix = model.agent_C.get_concept_matrix()
            loss, loss_bce, loss_align, loss_mono = criterion(
                predictions, labels, mask, states, attns, concept_matrix
            )

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                optimizer.step()

        total_loss += loss.item()
        total_bce += loss_bce.item()
        total_align += loss_align.item()
        total_mono += loss_mono.item()
        num_batches += 1

        all_labels.append(labels.cpu())
        all_preds.append(predictions.detach().cpu())
        all_masks.append(mask.cpu())

    y_true = torch.cat(all_labels, dim=0).numpy().reshape(-1)
    y_score = torch.cat(all_preds, dim=0).numpy().reshape(-1)
    mask_np = torch.cat(all_masks, dim=0).numpy().reshape(-1)
    metrics = evaluate_all(y_true, y_score, mask_np)

    metrics["loss"] = total_loss / max(num_batches, 1)
    metrics["bce"] = total_bce / max(num_batches, 1)
    metrics["align"] = total_align / max(num_batches, 1)
    metrics["mono"] = total_mono / max(num_batches, 1)

    return metrics["loss"], metrics


def log_metrics(phase: str, epoch: int, metrics: Dict[str, float], elapsed: float | None = None) -> None:
    msg = f"[{phase}] Epoch {epoch:02d}"
    msg += f" | Loss: {metrics['loss']:.5f}"
    msg += f" (BCE {metrics['bce']:.5f}, Align {metrics['align']:.5f}, Mono {metrics['mono']:.5f})"
    msg += f" | AUC: {metrics['auc']:.5f}"
    msg += f" | ACC: {metrics['acc']:.5f}"
    msg += f" | F1:  {metrics['f1']:.5f}"
    msg += f" | RMSE: {metrics['rmse']:.5f}"
    msg += f" | MAE: {metrics['mae']:.5f}"
    if elapsed is not None:
        msg += f" | Time: {elapsed:.2f}s"
    print(msg)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    print(f"  -> Checkpoint saved: {path}")


# ---------------------------------------------------------------------------
# Main training procedure
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    config = build_config(args)

    device = get_device(config.train.device)
    print(f"Using device: {device}")

    # --- Data ---------------------------------------------------------------
    print("Loading datasets...")
    data_info = create_dataloaders(config)
    train_loader = data_info["train"]
    val_loader = data_info["val"]
    test_loader = data_info["test"]
    print(
        f"Datasets loaded: train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}"
    )
    print(
        f"Vocab sizes: problem={data_info['problem_vocab_size']}, "
        f"skill_id={data_info['skill_id_vocab_size']}, "
        f"skill_text={data_info['skill_text_vocab_size']}"
    )

    # --- Model / optimiser / loss -------------------------------------------
    model = create_model(config, data_info).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    criterion = MAKTLoss(
        lambda_align=config.train.lambda_align,
        lambda_mono=config.train.lambda_mono,
        delta=config.train.delta,
    )
    optimizer = Adam(model.parameters(), lr=config.train.learning_rate)
    scheduler = StepLR(
        optimizer,
        step_size=config.train.lr_decay_step,
        gamma=config.train.lr_decay_gamma,
    )

    # --- Training loop --------------------------------------------------------
    best_val_auc = 0.0
    patience_counter = 0
    start_epoch = 1
    history: List[Dict[str, Dict[str, float]]] = []
    checkpoint_dir = Path(config.train.checkpoint_dir)
    log_dir = Path(config.train.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Resume from an existing checkpoint if available (keeps the prior best AUC)
    resume_path = checkpoint_dir / "best_model.pt"
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_auc = ckpt["metrics"].get("auc", 0.0)
        print(f"Resuming from checkpoint: epoch {ckpt['epoch']} (best val AUC {best_val_auc:.5f})")
        # Fast-forward the LR scheduler to the current epoch
        for _ in range(ckpt["epoch"]):
            scheduler.step()

    for epoch in range(start_epoch, config.train.epochs + 1):
        start_time = time.time()

        train_loss, train_metrics = run_epoch(
            model, criterion, train_loader, device, config.train.clip_grad, optimizer
        )
        val_loss, val_metrics = run_epoch(
            model, criterion, val_loader, device, config.train.clip_grad, optimizer=None
        )

        scheduler.step()
        elapsed = time.time() - start_time

        log_metrics("Train", epoch, train_metrics)
        log_metrics("Val  ", epoch, val_metrics, elapsed=elapsed)

        history.append({"train": train_metrics, "val": val_metrics})

        # Save the best checkpoint on the validation AUC
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_dir / "best_model.pt")
        else:
            patience_counter += 1

        # Early stopping (never before min_epochs)
        if epoch >= config.train.min_epochs and patience_counter >= config.train.early_stop_patience:
            print(
                f"Validation AUC did not improve for {config.train.early_stop_patience} epochs. "
                f"Early stopping at epoch {epoch}."
            )
            break

    # --- Training history -----------------------------------------------------
    history_path = log_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"Training history saved: {history_path}")

    # --- Test evaluation ------------------------------------------------------
    print("\nLoading best model and evaluating on the test set...")
    best_ckpt = torch.load(checkpoint_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_metrics = run_epoch(
        model, criterion, test_loader, device, config.train.clip_grad, optimizer=None
    )
    log_metrics("Test ", best_ckpt["epoch"], test_metrics)

    test_result_path = log_dir / "test_metrics.json"
    with open(test_result_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    print(f"Test metrics saved: {test_result_path}")


if __name__ == "__main__":
    main()
