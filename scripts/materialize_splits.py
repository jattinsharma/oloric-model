#!/usr/bin/env python3
"""
Materialize deterministic train/validation/test splits from the audited dataset.

Split parameters (from configs/dataset.yaml):
  - train_ratio:      0.8  → 384 examples
  - validation_ratio: 0.1  →  48 examples
  - test_ratio:       0.1  →  48 examples
  - shuffle:          True
  - seed:             42

IMPORTANT: Do NOT re-run this script unless the dataset checksum has been
re-verified. The output files are deterministic given the input file and seed.
"""
import hashlib
import json
import os
import random
import sys
from pathlib import Path

AUTHORITATIVE_SHA256_LF = "9DA9F5E9B45CA85F718B3E71FD8FB81881D46101D9815C7323411C30C7625921"
DATASET_PATH = Path("data/generated/enhanced_seed_data.jsonl")
SPLIT_DIR = Path("data/splits")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42


def sha256_lf(path: Path) -> str:
    raw = path.read_bytes()
    normalized = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest().upper()


def main():
    # Verify dataset integrity
    actual_sha = sha256_lf(DATASET_PATH)
    if actual_sha != AUTHORITATIVE_SHA256_LF:
        print(f"ERROR: Dataset checksum mismatch!")
        print(f"  Expected: {AUTHORITATIVE_SHA256_LF}")
        print(f"  Actual:   {actual_sha}")
        sys.exit(1)
    print(f"Dataset checksum verified: {actual_sha[:16]}...")

    # Load all examples
    examples = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    total = len(examples)
    print(f"Loaded {total} examples")

    # Compute split boundaries
    train_n = int(total * TRAIN_RATIO)
    val_n = int(total * VAL_RATIO)
    test_n = total - train_n - val_n
    assert train_n + val_n + test_n == total

    # Deterministic shuffle by seed
    rng = random.Random(SEED)
    indices = list(range(total))
    rng.shuffle(indices)

    train_indices = set(indices[:train_n])
    val_indices = set(indices[train_n : train_n + val_n])
    test_indices = set(indices[train_n + val_n :])

    # Verify no overlap
    assert len(train_indices & val_indices) == 0
    assert len(train_indices & test_indices) == 0
    assert len(val_indices & test_indices) == 0
    print("Split overlap check: PASS (0 shared examples across train/val/test)")

    # Write split files
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train.jsonl": [examples[i] for i in indices[:train_n]],
        "validation.jsonl": [examples[i] for i in indices[train_n : train_n + val_n]],
        "test.jsonl": [examples[i] for i in indices[train_n + val_n :]],
    }

    for filename, split_examples in split_map.items():
        out_path = SPLIT_DIR / filename
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in split_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Written {len(split_examples):3d} examples -> {out_path}")

    print()
    print("=" * 50)
    print("SPLIT SUMMARY")
    print("=" * 50)
    print(f"  Source dataset:          {DATASET_PATH}")
    print(f"  Source SHA-256 (LF):     {actual_sha}")
    print(f"  Random seed:             {SEED}")
    print(f"  Algorithm:               random.Random(seed).shuffle(range(N))")
    print(f"  Total examples:          {total}")
    print(f"  Train  (80%):            {train_n}")
    print(f"  Validation (10%):        {val_n}")
    print(f"  Test   (10%):            {test_n}")
    print(f"  Benchmark (held out):    20  (evaluation/benchmark.jsonl)")
    print(f"  Overlap (any pair):      0")
    print()
    print("Split files written to: data/splits/")


if __name__ == "__main__":
    main()
