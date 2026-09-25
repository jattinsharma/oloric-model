#!/usr/bin/env python3
"""
Test the Oloric training data pipeline without running full training.

Checks:
  1. Load real training examples from data/splits/train.jsonl
  2. Validate TrainingExample schema
  3. Convert TrainingExample -> dict
  4. Tokenize using OloricFormatter with a lightweight tokenizer
  5. Verify types, keys, shapes, and -100 prompt masking
  6. Build OloricTorchDataset
  7. Run data collation (default_data_collator) on a batch
  8. Construct Trainer with real train split and retrieve first batch
     without running full optimizer steps (Task 8 requirement)

Does NOT load the 4B model or run 288 optimizer steps.
"""

import json
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_dir = os.path.join(_repo_root, "src")
for _p in (_src_dir, _repo_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    DataCollatorForLanguageModeling,
)

from oloric.schemas.dataset import TrainingExample
from oloric.formatting import OloricFormatter
from oloric.training import OloricTorchDataset


TRAIN_JSONL = os.path.join(_repo_root, "data", "splits", "train.jsonl")
TOKENIZER_ID = "gpt2"  # Fast, CPU-safe tokenizer for unit tests
MAX_LENGTH = 256
N_EXAMPLES = 2


def load_n_examples(path: str, n: int) -> list:
    """Load the first n lines from a JSONL file as plain dicts."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if len(items) >= n:
                break
    return items


def main():
    print("=" * 60)
    print("OLORIC TRAINING DATA PIPELINE VERIFICATION")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load real training examples from train.jsonl
    # ------------------------------------------------------------------
    print(f"\n[1] Loading {N_EXAMPLES} examples from {TRAIN_JSONL}...")
    assert os.path.exists(TRAIN_JSONL), f"MISSING: {TRAIN_JSONL}"
    raw_dicts = load_n_examples(TRAIN_JSONL, N_EXAMPLES)
    assert len(raw_dicts) == N_EXAMPLES, f"Expected {N_EXAMPLES} examples, got {len(raw_dicts)}"
    print(f"    Loaded {len(raw_dicts)} raw dicts")

    # ------------------------------------------------------------------
    # 2. Validate TrainingExample schema
    # ------------------------------------------------------------------
    print("\n[2] Validating TrainingExample Pydantic schema...")
    te_objects = [TrainingExample(**d) for d in raw_dicts]
    for te in te_objects:
        assert te.id, "TrainingExample.id is empty"
        assert te.context, "TrainingExample.context is empty"
        assert te.target, "TrainingExample.target is empty"
    print(f"    {len(te_objects)} TrainingExample objects validated")

    # ------------------------------------------------------------------
    # 3. Convert TrainingExample -> dict
    # ------------------------------------------------------------------
    print("\n[3] Converting TrainingExample -> plain dicts...")
    example_dicts = [te.dict() for te in te_objects]
    print(f"    OK: {len(example_dicts)} dicts")

    # ------------------------------------------------------------------
    # 4. Setup tokenizer and OloricFormatter
    # ------------------------------------------------------------------
    print(f"\n[4] Loading tokenizer: {TOKENIZER_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"    Vocab size:    {tokenizer.vocab_size}")
    print(f"    Pad token id:  {tokenizer.pad_token_id}")
    print(f"    EOS token id:  {tokenizer.eos_token_id}")

    # ------------------------------------------------------------------
    # 5. Preprocessing & tokenization (Task 5 specifications)
    # ------------------------------------------------------------------
    print("\n[5] Running complete preprocessing/tokenization on single example...")
    formatter = OloricFormatter(tokenizer)
    features = formatter.tokenize_example(example_dicts[0], max_length=MAX_LENGTH)

    print(f"    Python type of processed example: {type(features)}")
    print(f"    Keys: {list(features.keys())}")
    print(f"    input_ids type:      {type(features['input_ids'])} of {type(features['input_ids'][0])}")
    print(f"    attention_mask type: {type(features['attention_mask'])} of {type(features['attention_mask'][0])}")
    print(f"    labels type:         {type(features['labels'])} of {type(features['labels'][0])}")
    print(f"    Sequence length:     {len(features['input_ids'])}")

    assert set(features.keys()) == {"input_ids", "attention_mask", "labels"}
    assert len(features["input_ids"]) == MAX_LENGTH
    assert len(features["attention_mask"]) == MAX_LENGTH
    assert len(features["labels"]) == MAX_LENGTH

    n_active = sum(1 for l in features["labels"] if l != -100)
    n_masked = sum(1 for l in features["labels"] if l == -100)
    print(f"    Active target labels (loss computed): {n_active}")
    print(f"    Masked prompt/pad labels (-100):      {n_masked}")
    assert n_active > 0, "No active target labels found"
    assert n_masked > 0, "No masked prompt/pad labels found"

    # ------------------------------------------------------------------
    # 6. Build OloricTorchDataset
    # ------------------------------------------------------------------
    print("\n[6] Building OloricTorchDataset...")
    dataset = OloricTorchDataset(example_dicts, formatter, max_length=MAX_LENGTH)
    assert len(dataset) == N_EXAMPLES
    item0 = dataset[0]
    assert isinstance(item0, dict)
    assert isinstance(item0["input_ids"], torch.Tensor)
    assert isinstance(item0["attention_mask"], torch.Tensor)
    assert isinstance(item0["labels"], torch.Tensor)
    print(f"    Dataset length:                {len(dataset)}")
    print(f"    item[0]['input_ids'].shape:      {item0['input_ids'].shape}")
    print(f"    item[0]['attention_mask'].shape: {item0['attention_mask'].shape}")
    print(f"    item[0]['labels'].shape:         {item0['labels'].shape}")

    # ------------------------------------------------------------------
    # 7. Collator verification on tiny batch of 2 examples
    # ------------------------------------------------------------------
    print("\n[7] Testing collation with default_data_collator...")
    collator = default_data_collator
    batch = collator([dataset[0], dataset[1]])
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch

    expected_shape = torch.Size([N_EXAMPLES, MAX_LENGTH])
    assert batch["input_ids"].shape == expected_shape
    assert batch["attention_mask"].shape == expected_shape
    assert batch["labels"].shape == expected_shape
    assert batch["input_ids"].dtype == torch.int64
    assert batch["labels"].dtype == torch.int64

    # Verify that default_data_collator preserved prompt label masking
    collated_active = sum(1 for l in batch["labels"][0].tolist() if l != -100)
    print(f"    Collated batch shape:        {batch['input_ids'].shape}")
    print(f"    Collated active labels:      {collated_active}")
    assert collated_active == n_active, (
        f"Collator modified active label count! expected {n_active}, got {collated_active}"
    )

    # Also verify DataCollatorForLanguageModeling handles dataset items without crash
    print("    Verifying DataCollatorForLanguageModeling handles dataset items without crash...")
    lm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    lm_batch = lm_collator([dataset[0], dataset[1]])
    assert lm_batch["input_ids"].shape == expected_shape
    print("    [PASS] No AttributeError: 'TrainingExample' object has no attribute 'size'")

    # ------------------------------------------------------------------
    # 8. Trainer initialization & first batch test (Task 8)
    # ------------------------------------------------------------------
    print("\n[8] Task 8: Trainer initialization and first batch retrieval...")
    output_temp_dir = os.path.join(_repo_root, "checkpoints", "test_trainer_preflight")
    os.makedirs(output_temp_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_temp_dir,
        per_device_train_batch_size=2,
        do_train=True,
        report_to="none",
        dataloader_pin_memory=False,
    )

    class MinimalCausalLM(nn.Module):
        """Minimal mock model to test Trainer dataloader pipeline without downloading 4B params."""
        def __init__(self, vocab_size: int = 50257):
            super().__init__()
            self.linear = nn.Linear(MAX_LENGTH, 1)
        def forward(self, input_ids=None, attention_mask=None, labels=None):
            loss = torch.tensor(1.0, requires_grad=True)
            return type("ModelOutput", (), {"loss": loss, "logits": torch.zeros((2, MAX_LENGTH, 50257))})()

    trainer = Trainer(
        model=MinimalCausalLM(),
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )

    # Retrieve first batch from Trainer's actual DataLoader
    train_dl = trainer.get_train_dataloader()
    first_batch = next(iter(train_dl))

    assert "input_ids" in first_batch
    assert "attention_mask" in first_batch
    assert "labels" in first_batch
    assert first_batch["input_ids"].shape == expected_shape
    assert first_batch["attention_mask"].shape == expected_shape
    assert first_batch["labels"].shape == expected_shape
    assert first_batch["input_ids"].dtype == torch.int64
    assert first_batch["labels"].dtype == torch.int64

    trainer_active = sum(1 for l in first_batch["labels"][0].tolist() if l != -100)
    print(f"    Trainer first batch keys:       {list(first_batch.keys())}")
    print(f"    Trainer first batch shape:      {first_batch['input_ids'].shape}")
    print(f"    Trainer first batch dtypes:     input_ids={first_batch['input_ids'].dtype}, labels={first_batch['labels'].dtype}")
    print(f"    Trainer active label positions: {trainer_active}")
    print("    [PASS] Trainer successfully obtained first batch without type errors!")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING_DATA_PIPELINE_VERIFIED = YES")
    print("All checks (Task 1 - Task 8) PASSED successfully.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
