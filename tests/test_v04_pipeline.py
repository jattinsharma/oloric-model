#!/usr/bin/env python3
"""
Regression and pipeline tests for OLORIC v0.4.

Verifies end-to-end preservation of:
  - 600-record dataset counts (480 train / 60 val / 60 test)
  - 20 canonical categories (document_grounded=120, error_correction=48, 18 other=24)
  - 6 domains * 100 records, 60 concepts * 10 records
  - Prior error correction pipeline (Capability 1)
  - Retrieved evidence prioritization pipeline (Capability 2, Pattern A & B)
  - Document absence / refusal to invent pipeline (Capability 3)
  - Prompt formatting, task/instruction preservation, and label masking
"""
import copy
import json
import os
import sys
from collections import Counter
from pathlib import Path
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_dir = os.path.join(_repo_root, "src")
for _p in (_src_dir, _repo_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoTokenizer
from oloric.schemas.dataset import TrainingExample
from oloric.schemas.model_input import OloricModelInput
from oloric.formatting import OloricFormatter, example_to_model_input

CHATML_TEMPLATE = (
    "{% for m in messages %}"
    "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.chat_template = CHATML_TEMPLATE
    return tok


@pytest.fixture
def formatter(tokenizer):
    return OloricFormatter(tokenizer)


@pytest.fixture(scope="module")
def v04_records():
    dataset_path = Path(_repo_root) / "data" / "generated" / "oloric_v04_dataset.jsonl"
    assert dataset_path.exists(), f"Missing dataset file: {dataset_path}"
    with open(dataset_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="module")
def v04_splits():
    splits_dir = Path(_repo_root) / "data" / "splits_v04"
    splits = {}
    for s in ["train", "validation", "test"]:
        p = splits_dir / f"{s}.jsonl"
        assert p.exists(), f"Missing split file: {p}"
        with open(p, "r", encoding="utf-8") as f:
            splits[s] = [json.loads(line) for line in f if line.strip()]
    return splits


def test_v04_split_counts_and_isolation(v04_splits):
    assert len(v04_splits["train"]) == 480
    assert len(v04_splits["validation"]) == 60
    assert len(v04_splits["test"]) == 60

    train_ids = {r["id"] for r in v04_splits["train"]}
    val_ids = {r["id"] for r in v04_splits["validation"]}
    test_ids = {r["id"] for r in v04_splits["test"]}

    assert len(train_ids & val_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0


def test_v04_category_and_domain_balance(v04_records):
    assert len(v04_records) == 600

    cats = Counter(r["category"] for r in v04_records)
    assert cats["document_grounded"] == 120
    assert cats["error_correction"] == 48
    for c, count in cats.items():
        if c not in ["document_grounded", "error_correction"]:
            assert count == 24, f"Category {c} expected 24, got {count}"

    doms = Counter(r["domain"] for r in v04_records)
    assert len(doms) == 6
    for d, count in doms.items():
        assert count == 100, f"Domain {d} expected 100, got {count}"


def test_v04_concept_coverage(v04_records):
    concepts = Counter(r["context"]["learner_state"]["concept"] for r in v04_records)
    assert len(concepts) == 60
    for c, count in concepts.items():
        assert count == 10, f"Concept '{c}' expected 10 records, got {count}"


def test_v04_id_identity(v04_records):
    ids = [r["id"] for r in v04_records]
    assert len(ids) == len(set(ids)) == 600
    assert all(i.startswith("v04_") for i in ids)
    assert not any("v03_" in i or "v02_" in i for i in ids)


def test_v04_prior_error_correction_pipeline(v04_records):
    prior_errors = [r for r in v04_records if r["category"] == "error_correction" and int(r["id"].split("_")[-1]) >= 5]
    assert len(prior_errors) == 24

    for r in prior_errors:
        turns = r["context"]["conversation_context"]
        assert len(turns) >= 3
        assert turns[1]["role"] == "tutor"
        assert turns[2]["role"] == "student"

        target_resp = r["target"]["response"].lower()
        assert any(w in target_resp for w in ["apologize", "error", "mistake", "misunderstanding"])

        diag = r["target"]["diagnosis"]
        assert diag["confusion_type"] == "prior_tutor_error"
        assert diag["severity"] == "high"

        mem = r["target"]["memory"]
        assert mem["candidate"] is True
        assert mem["memory_type"] == "error_correction"
        assert mem["confidence"] == 0.95


def test_v04_evidence_prioritization_pipeline(v04_records):
    ev_priority = [r for r in v04_records if r["category"] == "document_grounded" and 13 <= int(r["id"].split("_")[-1]) <= 16]
    assert len(ev_priority) == 24

    pattern_a = [r for r in ev_priority if int(r["id"].split("_")[-1]) in [13, 14]]
    pattern_b = [r for r in ev_priority if int(r["id"].split("_")[-1]) in [15, 16]]
    assert len(pattern_a) == 12
    assert len(pattern_b) == 12

    for r in pattern_a:
        assert r["target"]["diagnosis"]["confusion_type"] == "evidence_prioritization"
        assert len(r["context"]["document_context"]["retrieved_evidence"]) > 0

    for r in pattern_b:
        assert r["target"]["diagnosis"]["confusion_type"] == "mechanistic_derivation"
        assert len(r["context"]["document_context"]["retrieved_evidence"]) > 0


def test_v04_absence_refusal_pipeline(v04_records):
    refusals = [r for r in v04_records if r["category"] == "document_grounded" and 17 <= int(r["id"].split("_")[-1]) <= 20]
    assert len(refusals) == 24

    for r in refusals:
        resp = r["target"]["response"].lower()
        assert any(w in resp for w in ["no mention", "not mention", "does not contain"])
        assert any(w in resp for w in ["consult", "source", "resource", "needed"])

        diag = r["target"]["diagnosis"]
        assert diag["confusion_type"] == "document_absence"

        mem = r["target"]["memory"]
        assert mem["candidate"] is False
        assert mem["title"] is None
        assert mem["content"] is None
        assert mem["confidence"] == 0.0


def test_v04_prompt_formatting_preserves_task_and_instruction(formatter, v04_records):
    rec = v04_records[0]
    mi = example_to_model_input(rec)
    prompt = formatter.format_input(mi)

    assert mi.task in prompt
    assert mi.instruction in prompt
    assert mi.document_context.title in prompt
    assert mi.document_context.section in prompt
    for turn in mi.conversation_context:
        assert turn.content in prompt


def test_v04_tokenization_and_label_masking(formatter, v04_records):
    rec = v04_records[0]
    feats = formatter.tokenize_example(rec, max_length=512)
    assert "input_ids" in feats
    assert "labels" in feats

    labels = feats["labels"]
    input_ids = feats["input_ids"]

    active_indices = [i for i, l in enumerate(labels) if l != -100]
    assert len(active_indices) > 0
    # Active tokens match input_ids
    for i in active_indices:
        assert labels[i] == input_ids[i]

    # Prompt is masked
    first_active = active_indices[0]
    for i in range(first_active):
        assert labels[i] == -100
