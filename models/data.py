"""
Data loading for the MAKT model.

Supports the five benchmark datasets used in the paper:
    - ASSISTments2017   (2017 ASSISTments data mining competition)
    - Algebra2005-2006  (KDD Cup 2010)
    - Algebra2006-2007  (KDD Cup 2010)
    - MoocRadar         (coarse-grained mapping version)
    - XES3G5M           (question-level interactions)

Preprocessing follows the protocol described in the paper:
    1. Each question-answering event is treated as one interaction.
    2. Learners whose interaction sequences contain fewer than
       `min_records` (default 10) records are discarded.
    3. The remaining records of each learner are sorted chronologically
       to construct the learning sequence.
    4. Sequences are truncated to at most `seq_len` (default 200) time
       steps; the exceeding portions are split into new subsequences.
    5. 80% of the learners are used for training and 20% for
       validation and testing (10% / 10%).

Provides:
    - Vocabulary            : symbol vocabulary with <PAD> (0) / <UNK> (1) tokens
    - MAKTDataset           : fixed-length sequence dataset built from
                              per-student interaction records
    - collate_fn            : batch collation
    - prepare_csv_data      : ASSISTments2017 / Algebra2005-2006 /
                              Algebra2006-2007 preparation (shared CSV schema)
    - prepare_mooc_data     : MoocRadar train/val/test preparation
    - prepare_xes3g5m_data  : XES3G5M train/val/test preparation

File locations are configured in config.DataConfig. Built vocabularies are
cached as pickle files under DataConfig.cache_dir so that later runs skip
the vocabulary construction.
"""

from __future__ import annotations

import gc
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

DEFAULT_SEQ_LEN = 200
DEFAULT_MAX_TEXT_LEN = 20
DEFAULT_MIN_RECORDS = 10
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
RANDOM_SEED = 2024


# ---------------------------------------------------------------------------
# Vocabulary utilities
# ---------------------------------------------------------------------------
class Vocabulary:
    """Simple symbol vocabulary with padding (0) and unknown (1) tokens."""

    def __init__(
        self,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        special_tokens: Sequence[str] | None = None,
    ) -> None:
        self.pad_token = pad_token
        self.unk_token = unk_token
        self._token2idx: Dict[str, int] = {pad_token: 0, unk_token: 1}
        self._idx2token: Dict[int, str] = {0: pad_token, 1: unk_token}
        if special_tokens is not None:
            for token in special_tokens:
                self.add(token)

    @property
    def pad_idx(self) -> int:
        return self._token2idx[self.pad_token]

    @property
    def unk_idx(self) -> int:
        return self._token2idx[self.unk_token]

    def add(self, token: str) -> int:
        if token not in self._token2idx:
            idx = len(self._token2idx)
            self._token2idx[token] = idx
            self._idx2token[idx] = token
        return self._token2idx[token]

    def build(self, tokens: Sequence[str]) -> None:
        for token in tokens:
            self.add(token)

    def __getitem__(self, token: str) -> int:
        return self._token2idx.get(token, self.unk_idx)

    def get_token(self, idx: int) -> str:
        return self._idx2token.get(idx, self.unk_token)

    def __len__(self) -> int:
        return len(self._token2idx)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "pad_token": self.pad_token,
                    "unk_token": self.unk_token,
                    "token2idx": self._token2idx,
                    "idx2token": self._idx2token,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        with open(path, "rb") as f:
            data = pickle.load(f)
        vocab = cls(pad_token=data["pad_token"], unk_token=data["unk_token"])
        vocab._token2idx = data["token2idx"]
        vocab._idx2token = data["idx2token"]
        return vocab


def tokenize_skill_text(text: str) -> List[str]:
    """Tokenise a skill text, e.g. "Time; Measurement" -> ["time", "measurement"]."""
    cleaned = str(text).lower()
    for ch in r":,;()[]{}\/":
        cleaned = cleaned.replace(ch, " ")
    return [t.strip() for t in cleaned.replace("_", " ").replace("-", " ").split() if t.strip()]


# ---------------------------------------------------------------------------
# Common sequence utilities
# ---------------------------------------------------------------------------
def filter_short_sequences(
    student_records: Dict[str, List[Dict[str, object]]],
    min_records: int = DEFAULT_MIN_RECORDS,
) -> Dict[str, List[Dict[str, object]]]:
    """Discard learners whose interaction sequences contain fewer than
    `min_records` records (paper preprocessing protocol)."""
    return {
        sid: records
        for sid, records in student_records.items()
        if len(records) >= min_records
    }


def split_students_by_id(
    student_ids: Sequence[str],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Shuffle student ids and split them into train / val / test lists.

    With the paper defaults (train_ratio=0.8, val_ratio=0.1), 80% of the
    learners are used for training, 10% for validation and 10% for testing.
    """
    rng = random.Random(seed)
    ids = list(student_ids)
    rng.shuffle(ids)

    n_total = len(ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]
    return train_ids, val_ids, test_ids


def truncate_student_sequence(
    interactions: List[Dict[str, object]],
    seq_len: int = DEFAULT_SEQ_LEN,
) -> List[List[Dict[str, object]]]:
    """Split a student's interaction history into non-overlapping chunks of
    seq_len (the exceeding portions become new subsequences)."""
    chunks: List[List[Dict[str, object]]] = []
    for i in range(0, len(interactions), seq_len):
        chunk = interactions[i : i + seq_len]
        if chunk:
            chunks.append(chunk)
    return chunks


def build_vocabularies(
    student_records: Dict[str, List[Dict[str, object]]],
) -> Tuple[Vocabulary, Vocabulary, Vocabulary]:
    """
    Build the problem-id, skill-id and skill-text vocabularies from the
    per-student interaction records.

    Returns:
        (problem_vocab, skill_id_vocab, skill_text_vocab)
    """
    problem_tokens = sorted({r["problemId"] for records in student_records.values() for r in records})
    skill_id_tokens = sorted({r["skill_id"] for records in student_records.values() for r in records})

    skill_text_tokens: set[str] = set()
    for records in student_records.values():
        for r in records:
            skill_text_tokens.update(tokenize_skill_text(r["skill"]))
    skill_text_tokens = sorted(skill_text_tokens)

    problem_vocab = Vocabulary()
    problem_vocab.build(problem_tokens)

    skill_id_vocab = Vocabulary()
    skill_id_vocab.build(skill_id_tokens)

    skill_text_vocab = Vocabulary()
    skill_text_vocab.build(skill_text_tokens)

    return problem_vocab, skill_id_vocab, skill_text_vocab


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class MAKTDataset(Dataset):
    """
    Fixed-length-sequence dataset built from per-student interaction records.

    Each sample is a chunk of at most `seq_len` interactions, right-aligned
    in zero-padded tensors. `prev_correct[t]` is the response to the previous
    interaction (0 at t = 0), and `mask` marks the real positions.
    """

    def __init__(
        self,
        records: Dict[str, List[Dict[str, object]]],
        student_ids: Sequence[str],
        problem_vocab: Vocabulary,
        skill_id_vocab: Vocabulary,
        skill_text_vocab: Vocabulary,
        seq_len: int = DEFAULT_SEQ_LEN,
        max_text_len: int = DEFAULT_MAX_TEXT_LEN,
    ) -> None:
        self.problem_vocab = problem_vocab
        self.skill_id_vocab = skill_id_vocab
        self.skill_text_vocab = skill_text_vocab
        self.seq_len = seq_len
        self.max_text_len = max_text_len

        self.samples: List[List[Dict[str, object]]] = []
        for sid in student_ids:
            chunks = truncate_student_sequence(records[sid], seq_len=seq_len)
            self.samples.extend(chunks)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.samples[idx]
        length = len(chunk)

        problem_ids = torch.zeros(self.seq_len, dtype=torch.long)
        skill_ids = torch.zeros(self.seq_len, dtype=torch.long)
        skill_texts = torch.zeros(self.seq_len, self.max_text_len, dtype=torch.long)
        prev_correct = torch.zeros(self.seq_len, dtype=torch.float)
        labels = torch.zeros(self.seq_len, dtype=torch.float)
        mask = torch.zeros(self.seq_len, dtype=torch.float)

        for t, record in enumerate(chunk):
            problem_ids[t] = self.problem_vocab[record["problemId"]]
            skill_ids[t] = self.skill_id_vocab[record["skill_id"]]

            text_tokens = tokenize_skill_text(record["skill"])
            text_ids = [self.skill_text_vocab[tok] for tok in text_tokens]
            text_ids = text_ids[: self.max_text_len]
            if text_ids:
                skill_texts[t, : len(text_ids)] = torch.tensor(text_ids, dtype=torch.long)

            labels[t] = float(record["correct"])
            prev_correct[t] = float(chunk[t - 1]["correct"]) if t > 0 else 0.0
            mask[t] = 1.0

        return {
            "problem_ids": problem_ids,
            "skill_ids": skill_ids,
            "skill_texts": skill_texts,
            "prev_correct": prev_correct,
            "labels": labels,
            "mask": mask,
            "length": length,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack padded samples into a batch."""
    return {
        "problem_ids": torch.stack([b["problem_ids"] for b in batch]),
        "skill_ids": torch.stack([b["skill_ids"] for b in batch]),
        "skill_texts": torch.stack([b["skill_texts"] for b in batch]),
        "prev_correct": torch.stack([b["prev_correct"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "length": torch.tensor([b["length"] for b in batch], dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Shared preparation helpers
# ---------------------------------------------------------------------------
def _load_or_build_vocabularies(
    student_records: Dict[str, List[Dict[str, object]]],
    cache_dir: Path,
) -> Tuple[Vocabulary, Vocabulary, Vocabulary]:
    """Load the cached vocabularies if available, otherwise build and cache them."""
    vocab_cache = {
        "problem": cache_dir / "problem_vocab.pkl",
        "skill_id": cache_dir / "skill_id_vocab.pkl",
        "skill_text": cache_dir / "skill_text_vocab.pkl",
    }

    if all(p.exists() for p in vocab_cache.values()):
        problem_vocab = Vocabulary.load(vocab_cache["problem"])
        skill_id_vocab = Vocabulary.load(vocab_cache["skill_id"])
        skill_text_vocab = Vocabulary.load(vocab_cache["skill_text"])
    else:
        problem_vocab, skill_id_vocab, skill_text_vocab = build_vocabularies(student_records)
        problem_vocab.save(vocab_cache["problem"])
        skill_id_vocab.save(vocab_cache["skill_id"])
        skill_text_vocab.save(vocab_cache["skill_text"])

    return problem_vocab, skill_id_vocab, skill_text_vocab


def _make_dataset(
    records: Dict[str, List[Dict[str, object]]],
    student_ids: Sequence[str],
    vocabs: Tuple[Vocabulary, Vocabulary, Vocabulary],
    seq_len: int,
    max_text_len: int,
) -> MAKTDataset:
    problem_vocab, skill_id_vocab, skill_text_vocab = vocabs
    return MAKTDataset(
        records=records,
        student_ids=student_ids,
        problem_vocab=problem_vocab,
        skill_id_vocab=skill_id_vocab,
        skill_text_vocab=skill_text_vocab,
        seq_len=seq_len,
        max_text_len=max_text_len,
    )


def _prepare_from_records(
    student_records: Dict[str, List[Dict[str, object]]],
    seq_len: int,
    max_text_len: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    cache_dir: Path,
    min_records: int,
) -> Tuple[MAKTDataset, MAKTDataset, MAKTDataset, Vocabulary, Vocabulary, Vocabulary]:
    """Shared preparation pipeline (paper protocol):
    filter short sequences -> split learners 80/10/10 -> build vocabularies
    -> construct the fixed-length sequence datasets."""
    student_records = filter_short_sequences(student_records, min_records=min_records)

    train_ids, val_ids, test_ids = split_students_by_id(
        list(student_records.keys()),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    vocabs = _load_or_build_vocabularies(student_records, cache_dir)

    train_dataset = _make_dataset(student_records, train_ids, vocabs, seq_len, max_text_len)
    val_dataset = _make_dataset(student_records, val_ids, vocabs, seq_len, max_text_len)
    test_dataset = _make_dataset(student_records, test_ids, vocabs, seq_len, max_text_len)

    return (train_dataset, val_dataset, test_dataset, *vocabs)


# ---------------------------------------------------------------------------
# ASSISTments2017 / Algebra2005-2006 / Algebra2006-2007 loader
# ---------------------------------------------------------------------------
def load_csv_interactions(
    csv_path: str | Path,
    nrows: int | None = None,
) -> Dict[str, List[Dict[str, object]]]:
    """
    Read a preprocessed KT CSV file and group the interactions by student.

    Shared schema for ASSISTments2017, Algebra2005-2006 and Algebra2006-2007.
    Expected columns:
        user_id, problem_id, skill_id, is_correct, timestamp
    Optional column:
        skill_text  (skill name; if absent, the skill id is used as the text)

    Rows with a missing timestamp keep the original file order.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    usecols = ["user_id", "problem_id", "skill_id", "is_correct", "timestamp"]
    df = pd.read_csv(csv_path, nrows=nrows)
    has_skill_text = "skill_text" in df.columns
    if has_skill_text:
        usecols.append("skill_text")
    df = df[usecols]

    student_records: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for _, row in df.iterrows():
        skill_id = str(row["skill_id"])
        cleaned = {
            "studentId": str(row["user_id"]),
            "problemId": str(row["problem_id"]),
            "skill_id": skill_id,
            "skill": str(row["skill_text"]) if has_skill_text else skill_id,
            "correct": int(row["is_correct"]),
            "startTime": float(row["timestamp"]) if pd.notna(row["timestamp"]) else 0.0,
        }
        student_records[cleaned["studentId"]].append(cleaned)

    # Chronological order within each student
    for sid in student_records:
        student_records[sid].sort(key=lambda x: x["startTime"])

    del df
    gc.collect()
    return dict(student_records)


def prepare_csv_data(
    csv_path: str | Path,
    seq_len: int = DEFAULT_SEQ_LEN,
    max_text_len: int = DEFAULT_MAX_TEXT_LEN,
    min_records: int = DEFAULT_MIN_RECORDS,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = RANDOM_SEED,
    cache_dir: str | Path | None = None,
    nrows: int | None = None,
) -> Tuple[MAKTDataset, MAKTDataset, MAKTDataset, Vocabulary, Vocabulary, Vocabulary]:
    """
    Prepare the train / val / test datasets for ASSISTments2017,
    Algebra2005-2006 or Algebra2006-2007 (shared preprocessed CSV schema).
    """
    csv_path = Path(csv_path)
    cache_dir = Path(cache_dir) if cache_dir is not None else csv_path.parent / "makt_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    student_records = load_csv_interactions(csv_path, nrows=nrows)

    return _prepare_from_records(
        student_records, seq_len, max_text_len,
        train_ratio, val_ratio, seed, cache_dir, min_records,
    )


# ---------------------------------------------------------------------------
# MoocRadar loader
# ---------------------------------------------------------------------------
def load_mooc_interactions_from_csv(
    csv_path: str | Path,
    nrows: int | None = None,
) -> Dict[str, List[Dict[str, object]]]:
    """
    Read the MoocRadar coarse-grained CSV and group the interactions by student.

    Expected columns:
        user_id, problem_id, is_correct, skill_id
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"MoocRadar file not found: {csv_path}")

    dtypes = {"user_id": "int64", "problem_id": "int64", "is_correct": "int64", "skill_id": "int64"}
    usecols = ["user_id", "problem_id", "is_correct", "skill_id"]
    df = pd.read_csv(csv_path, usecols=usecols, dtype=dtypes, nrows=nrows)

    student_records: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for _, row in df.iterrows():
        cleaned = {
            "studentId": str(int(row["user_id"])),
            "problemId": str(int(row["problem_id"])),
            "skill_id": str(int(row["skill_id"])),
            "skill": str(int(row["skill_id"])),
            "correct": int(row["is_correct"]),
            # MoocRadar coarse.csv does not provide reliable timestamps;
            # the original file order is preserved.
            "startTime": 0.0,
        }
        student_records[cleaned["studentId"]].append(cleaned)

    del df
    gc.collect()
    return dict(student_records)


def prepare_mooc_data(
    csv_path: str | Path,
    seq_len: int = DEFAULT_SEQ_LEN,
    max_text_len: int = DEFAULT_MAX_TEXT_LEN,
    min_records: int = DEFAULT_MIN_RECORDS,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = RANDOM_SEED,
    cache_dir: str | Path | None = None,
    nrows: int | None = None,
) -> Tuple[MAKTDataset, MAKTDataset, MAKTDataset, Vocabulary, Vocabulary, Vocabulary]:
    """Prepare the MoocRadar train / val / test datasets (paper protocol)."""
    csv_path = Path(csv_path)
    cache_dir = Path(cache_dir) if cache_dir is not None else csv_path.parent / "makt_cache_mooc"
    cache_dir.mkdir(parents=True, exist_ok=True)

    student_records = load_mooc_interactions_from_csv(csv_path, nrows=nrows)

    return _prepare_from_records(
        student_records, seq_len, max_text_len,
        train_ratio, val_ratio, seed, cache_dir, min_records,
    )


# ---------------------------------------------------------------------------
# XES3G5M loader
# ---------------------------------------------------------------------------
def load_xes3g5m_interactions_from_csv(
    csv_path: str | Path,
    nrows: int | None = None,
) -> Dict[str, List[Dict[str, object]]]:
    """
    Read the XES3G5M question-level CSV and group the interactions by student.

    Expected columns:
        uid, questions, concepts, responses, timestamps, image2_id
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"XES3G5M file not found: {csv_path}")

    dtypes = {
        "uid": "int32",
        "questions": "int64",
        "concepts": "int32",
        "responses": "int32",
        "timestamps": "int64",
        "image2_id": "int64",
    }
    usecols = ["uid", "questions", "concepts", "responses", "timestamps", "image2_id"]
    df = pd.read_csv(csv_path, usecols=usecols, dtype=dtypes, nrows=nrows)

    student_records: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for _, row in df.iterrows():
        cleaned = {
            "studentId": str(int(row["uid"])),
            "problemId": str(int(row["questions"])),
            "skill_id": str(int(row["concepts"])),
            "skill": str(int(row["concepts"])),
            "correct": int(row["responses"]),
            "startTime": float(row["timestamps"]),
        }
        student_records[cleaned["studentId"]].append(cleaned)

    # Chronological order within each student
    for sid in student_records:
        student_records[sid].sort(key=lambda x: x["startTime"])

    del df
    gc.collect()
    return dict(student_records)


def prepare_xes3g5m_data(
    csv_path: str | Path,
    seq_len: int = DEFAULT_SEQ_LEN,
    max_text_len: int = DEFAULT_MAX_TEXT_LEN,
    min_records: int = DEFAULT_MIN_RECORDS,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = RANDOM_SEED,
    cache_dir: str | Path | None = None,
    nrows: int | None = None,
) -> Tuple[MAKTDataset, MAKTDataset, MAKTDataset, Vocabulary, Vocabulary, Vocabulary]:
    """Prepare the XES3G5M train / val / test datasets (paper protocol)."""
    csv_path = Path(csv_path)
    cache_dir = Path(cache_dir) if cache_dir is not None else csv_path.parent / "makt_cache_xes3g5m"
    cache_dir.mkdir(parents=True, exist_ok=True)

    student_records = load_xes3g5m_interactions_from_csv(csv_path, nrows=nrows)

    return _prepare_from_records(
        student_records, seq_len, max_text_len,
        train_ratio, val_ratio, seed, cache_dir, min_records,
    )
