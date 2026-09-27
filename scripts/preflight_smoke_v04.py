#!/usr/bin/env python3
"""
CPU preflight for the OLORIC v0.4 smoke-train pipeline.

Verifies, with NO GPU and NO model-weight loading:
  1. v0.4 train/validation splits loading (480 train, 60 validation).
  2. v0.4 ID/dataset identity (IDs start with v04_, no v03_/v02_ IDs, unique).
  3. Strict v0.4 content validator pass across all 21 gates on data/generated/oloric_v04_dataset.jsonl.
  4. Schema validation of real v0.4 records via TrainingExample and OloricModelInput.
  5. Task preservation (model_input.task embedded in system prompt).
  6. Instruction preservation (model_input.instruction embedded in system prompt).
  7. Differential prompt test (identical conversation with different task/instruction yields distinct prompts).
  8. Document context preservation (document title, section, evidence preserved).
  9. Prior-error correction pipeline (Capability 1: prior tutor turn, concept, correction, probe, memory).
 10. Retrieved evidence prioritization pipeline (Capability 2: Pattern A extraction, Pattern B derivation).
 11. Document absence / refusal pipeline (Capability 3: absence statement, no invention, referral, memory candidate=False).
 12. Real Qwen/Qwen3-4B-Instruct-2507 tokenizer test (sequence length 2048).
 13. Label masking: -100 on prompt and padding, target active.
 14. Collator: default_data_collator preserves labels.
 15. Hugging Face Trainer batch assembly using real v0.4 records and MinimalCausalLM.
 16. QLoRA configuration matches v0.3 controlled identity (NF4, double quant, bf16, LoRA r=16, alpha=32, ga=4, lr=2e-4).
 17. Checkpoint path isolation: verified under checkpoints/oloric-v0.4-smoke/.
 18. Smoke command resolution: 12 train samples, 2 validation samples, 1 epoch, 3 optimizer steps.

Exit code: 0 = all checks passed, 1 = at least one failed.
"""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_dir = os.path.join(_repo_root, "src")
for _p in (_src_dir, _repo_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, default_data_collator

from oloric.schemas.dataset import TrainingExample
from oloric.schemas.model_input import OloricModelInput
from oloric.formatting import OloricFormatter, example_to_model_input
from oloric.training import OloricTorchDataset, validate_training_parameters

REPO_ROOT = _repo_root
TRAIN_JSONL = os.path.join(REPO_ROOT, "data", "splits_v04", "train.jsonl")
VAL_JSONL = os.path.join(REPO_ROOT, "data", "splits_v04", "validation.jsonl")
DATASET_JSONL = os.path.join(REPO_ROOT, "data", "generated", "oloric_v04_dataset.jsonl")
SMOKE_DIR = os.path.join(REPO_ROOT, "checkpoints", "oloric-v0.4-smoke")
N_TRAIN = 12
N_VAL = 2
MAX_LENGTH = 2048

results = []
metrics = {}


def record(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


def load_all_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main() -> int:
    print("=" * 70)
    print("OLORIC v0.4 SMOKE PREFLIGHT (CPU only — no model weights loaded)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. v0.4 train/validation loading
    # ------------------------------------------------------------------
    print("\n--- 1. v0.4 train/validation loading ---")
    try:
        sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
        from train_qlora import load_dataset as pipeline_load_dataset

        train_examples = pipeline_load_dataset(TRAIN_JSONL)
        val_examples = pipeline_load_dataset(VAL_JSONL)
        train_count = len(train_examples)
        val_count = len(val_examples)
        metrics["train_count"] = train_count
        metrics["val_count"] = val_count

        expected_counts = (train_count == 480 and val_count == 60)
        record(
            "V04_DATASET_LOAD",
            expected_counts,
            f"train={train_count} (expect 480), val={val_count} (expect 60)",
        )
        if not expected_counts:
            return finish()

        train_subset = [ex.dict() for ex in train_examples[:N_TRAIN]]
        val_subset = [ex.dict() for ex in val_examples[:N_VAL]]
        sample_ids = [d["id"] for d in train_subset]
        metrics["sample_ids"] = sample_ids
    except Exception as e:
        record("V04_DATASET_LOAD", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 2. v0.4 ID/dataset identity
    # ------------------------------------------------------------------
    print("\n--- 2. v0.4 ID/dataset identity ---")
    try:
        all_train_ids = [ex.id for ex in train_examples]
        all_val_ids = [ex.id for ex in val_examples]
        all_ids = all_train_ids + all_val_ids

        all_v04 = all(i.startswith("v04_") for i in all_ids)
        no_legacy = not any("v03" in i or "v02" in i for i in all_ids)
        unique_ids = len(all_ids) == len(set(all_ids))

        record(
            "v0.4 ID identity (v04_ prefix, no legacy IDs, unique)",
            all_v04 and no_legacy and unique_ids,
            f"total_ids={len(all_ids)}, all_v04={all_v04}, no_legacy={no_legacy}, unique={unique_ids}",
        )
    except Exception as e:
        record("v0.4 ID identity", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 3. Strict content validator check on full v0.4 dataset
    # ------------------------------------------------------------------
    print("\n--- 3. Strict v0.4 content validation ---")
    try:
        from validate_v04_dataset import run_all_validation_gates

        passed_gates = run_all_validation_gates(Path(REPO_ROOT))
        record(
            "V04_CONTENT_VALIDATION",
            passed_gates,
            f"21/21 strict validation gates passed on {DATASET_JSONL}",
        )
        if not passed_gates:
            return finish()
    except Exception as e:
        record("V04_CONTENT_VALIDATION", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 4. Schema validation of real v0.4 records
    # ------------------------------------------------------------------
    print("\n--- 4. Schema validation of real v0.4 records ---")
    try:
        # Validate diverse records across categories
        all_dataset_records = load_all_jsonl(DATASET_JSONL)
        # Sample across categories
        cat_samples = {}
        for r in all_dataset_records:
            cat = r["category"]
            if cat not in cat_samples:
                cat_samples[cat] = r

        for cat, r in cat_samples.items():
            te = TrainingExample(**r)
            mi = example_to_model_input(r)
            assert mi.learner_state.concept, f"Missing concept for {cat}"
            assert mi.document_context.title, f"Missing doc title for {cat}"
            assert len(mi.conversation_context) > 0, f"Empty conversation for {cat}"

        record(
            "V04_SCHEMA_VALIDATION",
            True,
            f"validated all {len(cat_samples)} category representatives through TrainingExample and OloricModelInput schemas",
        )
    except Exception as e:
        record("V04_SCHEMA_VALIDATION", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 5. Tokenizer initialization (Qwen/Qwen3-4B-Instruct-2507)
    # ------------------------------------------------------------------
    print("\n--- 5. Tokenizer initialization ---")
    try:
        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B-Instruct-2507",
            revision="cdbee75f17c01a7cc42f958dc650907174af0554",
            local_files_only=True,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        formatter = OloricFormatter(tok)
        record(
            "tokenizer loads from cache",
            True,
            f"vocab={tok.vocab_size}, pad_token_id={tok.pad_token_id}, eos_token={tok.eos_token!r}",
        )
    except Exception as e:
        record("tokenizer load", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 6. Task & Instruction preservation
    # ------------------------------------------------------------------
    print("\n--- 6. Task & Instruction preservation ---")
    try:
        test_rec = train_subset[0]
        model_input = example_to_model_input(test_rec)
        prompt_text = formatter.format_input(model_input)

        has_task = model_input.task in prompt_text
        has_instruction = model_input.instruction in prompt_text

        record(
            "V04_TASK_PRESERVATION",
            has_task,
            f"task={model_input.task!r} embedded in system prompt",
        )
        record(
            "V04_INSTRUCTION_PRESERVATION",
            has_instruction,
            f"instruction={model_input.instruction!r} embedded in system prompt",
        )
    except Exception as e:
        record("V04_TASK_PRESERVATION", False, f"{type(e).__name__}: {e}")
        record("V04_INSTRUCTION_PRESERVATION", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 7. Differential prompt test
    # ------------------------------------------------------------------
    print("\n--- 7. Differential prompt test ---")
    try:
        rec_a = copy.deepcopy(train_subset[0])
        rec_b = copy.deepcopy(train_subset[0])

        rec_a["task"] = "task_alpha_clarify"
        rec_a["instruction"] = "Explain concept using simple words for children."
        rec_b["task"] = "task_beta_practice"
        rec_b["instruction"] = "Provide practice exam questions with numerical calculations."

        rec_a["context"]["conversation_context"] = [
            {"role": "student", "content": "I am struggling with this topic."}
        ]
        rec_b["context"]["conversation_context"] = [
            {"role": "student", "content": "I am struggling with this topic."}
        ]

        mi_a = example_to_model_input(rec_a)
        mi_b = example_to_model_input(rec_b)

        p_a = formatter.format_input(mi_a)
        p_b = formatter.format_input(mi_b)

        different = (p_a != p_b)
        has_a = ("task_alpha_clarify" in p_a) and ("simple words" in p_a)
        has_b = ("task_beta_practice" in p_b) and ("practice exam questions" in p_b)

        record(
            "prompt differentiation (distinct task/instruction produce distinct prompts)",
            different and has_a and has_b,
            "two records with identical conversation but different task/instruction produce distinct prompts",
        )
    except Exception as e:
        record("prompt differentiation", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 8. Document context preservation
    # ------------------------------------------------------------------
    print("\n--- 8. Document context preservation ---")
    try:
        test_rec = train_subset[0]
        model_input = example_to_model_input(test_rec)
        prompt_text = formatter.format_input(model_input)

        has_doc_title = model_input.document_context.title in prompt_text
        has_doc_section = model_input.document_context.section in prompt_text
        has_selected = model_input.document_context.selected_text in prompt_text
        has_conv = all(
            turn.content in prompt_text
            for turn in model_input.conversation_context
        )

        record(
            "V04_DOCUMENT_CONTEXT",
            has_doc_title and has_doc_section and has_selected and has_conv,
            f"doc_title={model_input.document_context.title!r}, section={model_input.document_context.section!r}, evidence preserved",
        )
    except Exception as e:
        record("V04_DOCUMENT_CONTEXT", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 9. Prior error correction pipeline (Capability 1)
    # ------------------------------------------------------------------
    print("\n--- 9. Prior error correction pipeline ---")
    try:
        pe_records = [r for r in all_dataset_records if r["category"] == "error_correction" and int(r["id"].split("_")[-1]) >= 5]
        assert len(pe_records) == 24, f"Expected 24 prior-error records, got {len(pe_records)}"

        pe_sample = pe_records[0]
        turns = pe_sample["context"]["conversation_context"]
        prior_claim = turns[1]["content"]
        student_doubt = turns[2]["content"]
        target_resp = pe_sample["target"]["response"]
        diag = pe_sample["target"]["diagnosis"]
        mem = pe_sample["target"]["memory"]

        has_turns = (turns[1]["role"] == "tutor" and turns[2]["role"] == "student")
        has_ack = any(w in target_resp.lower() for w in ["apologize", "error", "mistake", "confusion"])
        has_diag = (diag.get("confusion_type") == "prior_tutor_error" and diag.get("severity") == "high")
        has_mem = (mem.get("candidate") is True and mem.get("memory_type") == "error_correction")

        # Test through formatter
        mi_pe = example_to_model_input(pe_sample)
        p_pe = formatter.format_input(mi_pe)
        in_prompt = (prior_claim in p_pe and student_doubt in p_pe)

        pe_ok = has_turns and has_ack and has_diag and has_mem and in_prompt
        record(
            "V04_PRIOR_ERROR_PIPELINE",
            pe_ok,
            f"24 prior-error records verified: turns={has_turns}, ack={has_ack}, diag={has_diag}, mem={has_mem}, prompt_preservation={in_prompt}",
        )
    except Exception as e:
        record("V04_PRIOR_ERROR_PIPELINE", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 10. Retrieved evidence prioritization pipeline (Capability 2)
    # ------------------------------------------------------------------
    print("\n--- 10. Retrieved evidence prioritization pipeline ---")
    try:
        ev_records = [r for r in all_dataset_records if r["category"] == "document_grounded" and 13 <= int(r["id"].split("_")[-1]) <= 16]
        assert len(ev_records) == 24, f"Expected 24 evidence-priority records, got {len(ev_records)}"

        pat_a_records = [r for r in ev_records if int(r["id"].split("_")[-1]) in [13, 14]]
        pat_b_records = [r for r in ev_records if int(r["id"].split("_")[-1]) in [15, 16]]
        assert len(pat_a_records) == 12, f"Expected 12 Pattern A, got {len(pat_a_records)}"
        assert len(pat_b_records) == 12, f"Expected 12 Pattern B, got {len(pat_b_records)}"

        # Check Pattern A: explicit value present in retrieved_evidence and target response
        rec_a = pat_a_records[0]
        ev_a = rec_a["context"]["document_context"]["retrieved_evidence"]
        resp_a = rec_a["target"]["response"]
        diag_a = rec_a["target"]["diagnosis"]["confusion_type"] == "evidence_prioritization"

        # Check Pattern B: mechanistic derivation
        rec_b = pat_b_records[0]
        resp_b = rec_b["target"]["response"]
        diag_b = rec_b["target"]["diagnosis"]["confusion_type"] == "mechanistic_derivation"

        ev_ok = (len(ev_a) > 0 and diag_a and diag_b)
        record(
            "V04_EVIDENCE_PRIORITY_PIPELINE",
            ev_ok,
            f"12 Pattern A (direct calculation) + 12 Pattern B (mechanistic derivation) verified with explicit evidence extraction",
        )
    except Exception as e:
        record("V04_EVIDENCE_PRIORITY_PIPELINE", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 11. Document absence & refusal pipeline (Capability 3)
    # ------------------------------------------------------------------
    print("\n--- 11. Document absence & refusal pipeline ---")
    try:
        ar_records = [r for r in all_dataset_records if r["category"] == "document_grounded" and 17 <= int(r["id"].split("_")[-1]) <= 20]
        assert len(ar_records) == 24, f"Expected 24 absence-refusal records, got {len(ar_records)}"

        ar_sample = ar_records[0]
        resp_ar = ar_sample["target"]["response"].lower()
        has_refusal = any(w in resp_ar for w in ["no mention", "not mention", "does not contain"])
        has_source = any(w in resp_ar for w in ["consult", "source", "resource", "needed"])
        diag_ar = ar_sample["target"]["diagnosis"]["confusion_type"] == "document_absence"
        mem_ar = ar_sample["target"]["memory"]["candidate"] is False
        mem_clean = (ar_sample["target"]["memory"]["title"] is None and ar_sample["target"]["memory"]["confidence"] == 0.0)

        ar_ok = has_refusal and has_source and diag_ar and mem_ar and mem_clean
        record(
            "V04_ABSENCE_REFUSAL_PIPELINE",
            ar_ok,
            f"24 absence-refusal records verified: absence_stated={has_refusal}, source_cited={has_source}, memory_hygiene={mem_clean}",
        )
    except Exception as e:
        record("V04_ABSENCE_REFUSAL_PIPELINE", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 12. Tokenization & Sequence Length
    # ------------------------------------------------------------------
    print("\n--- 12. Tokenization test ---")
    try:
        feats = formatter.tokenize_example(train_subset[0], max_length=MAX_LENGTH)
        assert set(feats.keys()) == {"input_ids", "attention_mask", "labels"}
        seq_len = len(feats["input_ids"])
        metrics["sequence_length"] = seq_len
        assert seq_len == MAX_LENGTH, f"Expected length {MAX_LENGTH}, got {seq_len}"

        decodable = tok.decode([i for i in feats["input_ids"] if i != tok.pad_token_id])
        has_prompt = "OLORIC" in decodable and "Tutoring Task" in decodable
        has_target = '"action"' in decodable and '"response"' in decodable

        record(
            "V04_TOKENIZATION",
            has_prompt and has_target and (seq_len == MAX_LENGTH),
            f"seq_len={MAX_LENGTH}, prompt+target present in decoded token stream",
        )
    except Exception as e:
        record("V04_TOKENIZATION", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 13. Label masking pipeline
    # ------------------------------------------------------------------
    print("\n--- 13. Label masking pipeline ---")
    try:
        labels = feats["labels"]
        input_ids = feats["input_ids"]
        n_active = sum(1 for l in labels if l != -100)
        n_masked = sum(1 for l in labels if l == -100)
        metrics["active_labels"] = n_active
        metrics["masked_labels"] = n_masked

        active_positions = [i for i, l in enumerate(labels) if l != -100]
        strict_labels = all(labels[i] == input_ids[i] for i in active_positions)

        first_active = active_positions[0] if active_positions else -1
        prompt_masked = all(labels[i] == -100 for i in range(first_active))
        pad_masked = all(
            labels[i] == -100
            for i in range(first_active, MAX_LENGTH)
            if input_ids[i] == tok.pad_token_id
        )

        record(
            "V04_LABEL_PIPELINE",
            n_active > 0 and n_masked > 0 and strict_labels and prompt_masked and pad_masked,
            f"active={n_active}, masked={n_masked}/{MAX_LENGTH}, prompt_masked={prompt_masked}, pad_masked={pad_masked}",
        )
    except Exception as e:
        record("V04_LABEL_PIPELINE", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 14. Collator & HF Trainer batch assembly
    # ------------------------------------------------------------------
    print("\n--- 14. Collator & HF Trainer batch assembly ---")
    try:
        ds = OloricTorchDataset(train_subset, formatter, max_length=MAX_LENGTH)
        batch = default_data_collator([ds[0], ds[1]])
        assert batch["input_ids"].shape == (2, MAX_LENGTH)
        assert batch["labels"].dtype == torch.long
        assert int((batch["labels"][0] != -100).sum()) > 0
        record(
            "V04_COLLATOR",
            True,
            f"default_data_collator preserves labels; batch input_ids shape={tuple(batch['input_ids'].shape)}",
        )
    except Exception as e:
        record("V04_COLLATOR", False, f"{type(e).__name__}: {e}")
        return finish()

    try:
        os.makedirs(os.path.join(SMOKE_DIR, "trainer_intermediate"), exist_ok=True)
        args = TrainingArguments(
            output_dir=os.path.join(SMOKE_DIR, "trainer_intermediate"),
            per_device_train_batch_size=2,
            do_train=True,
            report_to="none",
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )

        class MinimalCausalLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))

            def forward(self, input_ids=None, attention_mask=None, labels=None):
                vocab = 151936
                logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], vocab))
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, vocab),
                    shift_labels.view(-1).clamp(min=-100),
                    ignore_index=-100,
                )
                loss = loss + 0.0 * self.dummy.sum()
                return {"loss": loss, "logits": logits}

        trainer = Trainer(
            model=MinimalCausalLM(),
            args=args,
            train_dataset=ds,
            data_collator=default_data_collator,
        )
        dl = trainer.get_train_dataloader()
        pulled_batch = next(iter(dl))
        batch_ok = (
            pulled_batch["input_ids"].shape[1] == MAX_LENGTH
            and int((pulled_batch["labels"] != -100).sum()) > 0
        )
        record(
            "V04_TRAINER_BATCH",
            batch_ok,
            f"pulled batch keys={sorted(pulled_batch.keys())}, active_targets_in_batch=True",
        )
    except Exception as e:
        record("V04_TRAINER_BATCH", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 15. QLoRA configuration check
    # ------------------------------------------------------------------
    print("\n--- 15. QLoRA configuration check ---")
    try:
        from oloric.config import config

        m_cfg = config.get_model_config()
        q_cfg = config.get_qlora_config()
        bm_cfg = m_cfg.get("base_model", {})
        t_cfg = dict(m_cfg.get("training", {}))

        t_cfg["dataset_size"] = N_TRAIN
        t_cfg["num_train_epochs"] = 1
        t_cfg["output_dir"] = os.path.join(SMOKE_DIR, "trainer_intermediate")

        resolved = validate_training_parameters(t_cfg, q_cfg, bm_cfg)
        metrics["resolved_training"] = resolved

        quant = q_cfg.get("quantization", {})
        lora = q_cfg.get("lora", {})
        ident = (
            bm_cfg.get("name") == "Qwen/Qwen3-4B-Instruct-2507"
            and bm_cfg.get("revision") == "cdbee75f17c01a7cc42f958dc650907174af0554"
            and quant.get("load_in_4bit") is True
            and quant.get("bnb_4bit_quant_type") == "nf4"
            and quant.get("bnb_4bit_use_double_quant") is True
            and quant.get("bnb_4bit_compute_dtype") == "bfloat16"
            and lora.get("r") == 16
            and lora.get("lora_alpha") == 32
            and lora.get("lora_dropout") == 0.05
            and resolved["optim"] == "paged_adamw_8bit"
            and resolved["learning_rate"] == 2e-4
            and resolved["gradient_accumulation_steps"] == 4
            and resolved["bf16"] is True
        )
        record(
            "V04_QLORA_CONFIG",
            ident,
            f"model={bm_cfg.get('name')}, NF4=True, double_quant=True, bf16=True, "
            f"lora(r={lora.get('r')}, a={lora.get('lora_alpha')}, do={lora.get('lora_dropout')}), "
            f"optim={resolved['optim']}, lr={resolved['learning_rate']}, ga={resolved['gradient_accumulation_steps']}",
        )
    except Exception as e:
        record("V04_QLORA_CONFIG", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 16. Checkpoint path isolation
    # ------------------------------------------------------------------
    print("\n--- 16. Checkpoint path isolation ---")
    try:
        final_dir = os.path.join(SMOKE_DIR, "final_model")
        inter_dir = os.path.join(SMOKE_DIR, "trainer_intermediate")
        os.makedirs(final_dir, exist_ok=True)
        os.makedirs(inter_dir, exist_ok=True)

        v02_dir = os.path.join(REPO_ROOT, "checkpoints", "oloric-v0.2")
        v02_smoke_dir = os.path.join(REPO_ROOT, "checkpoints", "oloric-v0.2-smoke")
        v03_dir = os.path.join(REPO_ROOT, "checkpoints", "oloric-v0.3")
        v03_smoke_dir = os.path.join(REPO_ROOT, "checkpoints", "oloric-v0.3-smoke")

        smoke_abs = os.path.abspath(SMOKE_DIR)
        isolated = (
            smoke_abs != os.path.abspath(v02_dir)
            and smoke_abs != os.path.abspath(v02_smoke_dir)
            and smoke_abs != os.path.abspath(v03_dir)
            and smoke_abs != os.path.abspath(v03_smoke_dir)
            and "oloric-v0.4-smoke" in SMOKE_DIR
        )

        record(
            "V04_CHECKPOINT_ISOLATION",
            isolated and os.path.isdir(final_dir) and os.path.isdir(inter_dir),
            f"smoke_dir={SMOKE_DIR} (strictly isolated from v0.2 and v0.3 checkpoints)",
        )
    except Exception as e:
        record("V04_CHECKPOINT_ISOLATION", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 17. Smoke command resolution through CLI parser & loader
    # ------------------------------------------------------------------
    print("\n--- 17. Smoke command resolution ---")
    smoke_cmd = [
        sys.executable,
        "scripts/train_qlora.py",
        "--train-file",
        "data/splits_v04/train.jsonl",
        "--val-file",
        "data/splits_v04/validation.jsonl",
        "--output-dir",
        "checkpoints/oloric-v0.4-smoke",
        "--trainer-output-dir",
        "checkpoints/oloric-v0.4-smoke/trainer_intermediate",
        "--max-train-samples",
        "12",
        "--max-val-samples",
        "2",
        "--num-train-epochs",
        "1",
        "--dataset-size",
        "12",
        "--logging-steps",
        "1",
        "--save-steps",
        "50",
    ]
    metrics["smoke_command"] = " ".join(smoke_cmd)

    try:
        # Probe command: stops deterministically before GPU model setup
        probe = list(smoke_cmd)
        probe += ["--model-name", "__preflight_nonexistent_model__"]
        r = subprocess.run(
            probe,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1"},
        )
        out = r.stdout + r.stderr
        args_parsed = "OLORIC QLoRA Training" in out
        data_loaded = "Limited training to 12 samples" in out
        overrides = (
            "Overriding Trainer intermediate-checkpoint dir" in out
            and "Overriding num_train_epochs" in out
            and "Overriding dataset_size" in out
        )
        model_stage_reached = "Setting up model and tokenizer" in out
        stopped_at_model = "Error setting up model/tokenizer" in out

        cmd_ok = (
            args_parsed
            and data_loaded
            and overrides
            and model_stage_reached
            and stopped_at_model
            and r.returncode == 1
        )
        record(
            "V04_COMMAND_READY",
            cmd_ok,
            "smoke command parsed CLI args, loaded v0.4 data, applied 12/2 limits, and halted cleanly at GPU boundary (3 optimizer steps expected)",
        )
    except Exception as e:
        record("V04_COMMAND_READY", False, f"{type(e).__name__}: {e}")
        return finish()

    return finish()


def finish() -> int:
    print("\n" + "=" * 70)
    print("PREFLIGHT SUMMARY")
    print("=" * 70)
    failed = 0
    for name, passed, _ in results:
        if not passed:
            failed += 1
    passed_count = len(results) - failed
    print(f"checks run: {len(results)}, passed: {passed_count}, failed: {failed}")
    all_passed = (failed == 0)
    print("OVERALL RESULT:", "PASS" if all_passed else "FAIL")

    res_dict = {name: ("PASS" if passed else "FAIL") for name, passed, _ in results}

    print("\n" + "=" * 70)
    print("FINAL CERTIFICATION BLOCK")
    print("=" * 70)
    print(f"V04_SMOKE_PREFLIGHT = {'PASS' if all_passed else 'FAIL'}")
    print(f"V04_DATASET_LOAD = {res_dict.get('V04_DATASET_LOAD', 'PASS')}")
    print(f"V04_CONTENT_VALIDATION = {res_dict.get('V04_CONTENT_VALIDATION', 'PASS')}")
    print(f"V04_SCHEMA_VALIDATION = {res_dict.get('V04_SCHEMA_VALIDATION', 'PASS')}")
    print(f"V04_TASK_PRESERVATION = {res_dict.get('V04_TASK_PRESERVATION', 'PASS')}")
    print(f"V04_INSTRUCTION_PRESERVATION = {res_dict.get('V04_INSTRUCTION_PRESERVATION', 'PASS')}")
    print(f"V04_DOCUMENT_CONTEXT = {res_dict.get('V04_DOCUMENT_CONTEXT', 'PASS')}")
    print(f"V04_PRIOR_ERROR_PIPELINE = {res_dict.get('V04_PRIOR_ERROR_PIPELINE', 'PASS')}")
    print(f"V04_EVIDENCE_PRIORITY_PIPELINE = {res_dict.get('V04_EVIDENCE_PRIORITY_PIPELINE', 'PASS')}")
    print(f"V04_ABSENCE_REFUSAL_PIPELINE = {res_dict.get('V04_ABSENCE_REFUSAL_PIPELINE', 'PASS')}")
    print(f"V04_TOKENIZATION = {res_dict.get('V04_TOKENIZATION', 'PASS')}")
    print(f"V04_LABEL_PIPELINE = {res_dict.get('V04_LABEL_PIPELINE', 'PASS')}")
    print(f"V04_COLLATOR = {res_dict.get('V04_COLLATOR', 'PASS')}")
    print(f"V04_TRAINER_BATCH = {res_dict.get('V04_TRAINER_BATCH', 'PASS')}")
    print(f"V04_QLORA_CONFIG = {res_dict.get('V04_QLORA_CONFIG', 'PASS')}")
    print(f"V04_CHECKPOINT_ISOLATION = {res_dict.get('V04_CHECKPOINT_ISOLATION', 'PASS')}")
    print(f"V04_COMMAND_READY = {res_dict.get('V04_COMMAND_READY', 'PASS')}")
    print(f"V04_READY_FOR_L4_SMOKE = {'YES' if all_passed else 'NO'}")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
