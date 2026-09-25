#!/usr/bin/env python3
"""
Pytest wrapper for the Oloric training data pipeline and Trainer batch verification.
"""

import json
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_dir = os.path.join(_repo_root, "src")
for _p in (_src_dir, _repo_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest
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
TOKENIZER_ID = "gpt2"
MAX_LENGTH = 256
N_EXAMPLES = 2


def load_n_examples(path: str, n: int) -> list:
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


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def raw_dicts():
    assert os.path.exists(TRAIN_JSONL), f"MISSING: {TRAIN_JSONL}"
    return load_n_examples(TRAIN_JSONL, N_EXAMPLES)


@pytest.fixture(scope="module")
def example_dicts(raw_dicts):
    te_objects = [TrainingExample(**d) for d in raw_dicts]
    return [te.dict() for te in te_objects]


@pytest.fixture(scope="module")
def formatter(tokenizer):
    return OloricFormatter(tokenizer)


@pytest.fixture(scope="module")
def dataset(example_dicts, formatter):
    return OloricTorchDataset(example_dicts, formatter, max_length=MAX_LENGTH)


class TestTokenization:
    def test_raw_dicts_loaded(self, raw_dicts):
        assert len(raw_dicts) == N_EXAMPLES

    def test_training_example_schema(self, raw_dicts):
        for d in raw_dicts:
            te = TrainingExample(**d)
            assert te.id
            assert te.context
            assert te.target

    def test_tokenize_example_keys(self, formatter, example_dicts):
        features = formatter.tokenize_example(example_dicts[0], max_length=MAX_LENGTH)
        assert set(features.keys()) == {"input_ids", "attention_mask", "labels"}

    def test_tokenize_example_types(self, formatter, example_dicts):
        features = formatter.tokenize_example(example_dicts[0], max_length=MAX_LENGTH)
        assert isinstance(features["input_ids"], list)
        assert isinstance(features["attention_mask"], list)
        assert isinstance(features["labels"], list)

    def test_tokenize_example_length(self, formatter, example_dicts):
        features = formatter.tokenize_example(example_dicts[0], max_length=MAX_LENGTH)
        assert len(features["input_ids"]) == MAX_LENGTH
        assert len(features["attention_mask"]) == MAX_LENGTH
        assert len(features["labels"]) == MAX_LENGTH

    def test_label_masking_has_active_targets(self, formatter, example_dicts):
        features = formatter.tokenize_example(example_dicts[0], max_length=MAX_LENGTH)
        n_active = sum(1 for l in features["labels"] if l != -100)
        n_masked = sum(1 for l in features["labels"] if l == -100)
        assert n_active > 0, "All labels are -100; label masking is broken"
        assert n_masked > 0, "No prompt tokens masked to -100"


class TestOloricTorchDataset:
    def test_dataset_length(self, dataset):
        assert len(dataset) == N_EXAMPLES

    def test_dataset_item_type(self, dataset):
        item = dataset[0]
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_dataset_item_tensors(self, dataset):
        item = dataset[0]
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)

    def test_dataset_item_shape(self, dataset):
        item = dataset[0]
        expected = torch.Size([MAX_LENGTH])
        assert item["input_ids"].shape == expected
        assert item["attention_mask"].shape == expected
        assert item["labels"].shape == expected


class TestDataCollator:
    def test_default_data_collator_preserves_labels(self, dataset):
        batch = default_data_collator([dataset[0], dataset[1]])
        expected = torch.Size([N_EXAMPLES, MAX_LENGTH])
        assert batch["input_ids"].shape == expected
        assert batch["attention_mask"].shape == expected
        assert batch["labels"].shape == expected
        assert batch["input_ids"].dtype == torch.long
        assert batch["labels"].dtype == torch.long

        # Check prompt label masking (-100) is preserved
        active_sample = sum(1 for l in batch["labels"][0].tolist() if l != -100)
        assert active_sample > 0

    def test_dataloader_batch_no_crash(self, dataset, tokenizer):
        """Verify the previous AttributeError: 'TrainingExample' object has no attribute 'size' is resolved."""
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        loader = DataLoader(dataset, batch_size=N_EXAMPLES, collate_fn=collator)
        batch = next(iter(loader))
        assert "input_ids" in batch
        expected = torch.Size([N_EXAMPLES, MAX_LENGTH])
        assert batch["input_ids"].shape == expected


class TestTrainerInitializationAndBatch:
    def test_trainer_first_batch(self, dataset, tmp_path):
        """Task 8: Trainer constructs real data path and successfully obtains one batch."""
        args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            do_train=True,
            report_to="none",
            dataloader_pin_memory=False,
        )

        class MinimalCausalLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(MAX_LENGTH, 1)
            def forward(self, input_ids=None, attention_mask=None, labels=None):
                loss = torch.tensor(1.0, requires_grad=True)
                return type("ModelOutput", (), {"loss": loss, "logits": torch.zeros((2, MAX_LENGTH, 50257))})()

        trainer = Trainer(
            model=MinimalCausalLM(),
            args=args,
            train_dataset=dataset,
            data_collator=default_data_collator,
        )

        dl = trainer.get_train_dataloader()
        batch = next(iter(dl))
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape == torch.Size([2, MAX_LENGTH])
        assert batch["labels"].shape == torch.Size([2, MAX_LENGTH])
