#!/usr/bin/env python3
"""
CPU preflight for the OLORIC v0.3 smoke-train pipeline.

Verifies, with NO GPU and NO model-weight loading:
  1. v0.3 train/validation splits loading (384 train, 48 validation).
  2. v0.3 ID/dataset identity (IDs start with v03_, no v02_ IDs).
  3. Strict v0.3 content validator pass on data/generated/oloric_v03_dataset.jsonl.
  4. Schema validation of real v0.3 records via TrainingExample and OloricModelInput.
  5. Real v0.3 prompt construction (task, instruction, conversation, document preserved).
  6. Differential prompt test: identical conversation with different task/instruction yields different prompts.
  7. Real Qwen/Qwen3-4B-Instruct-2507 tokenizer test (sequence length 2048).
  8. Label masking: -100 on prompt and padding, target active.
  9. Collator: default_data_collator preserves labels.
  10. Hugging Face Trainer batch assembly using real v0.3 records and MinimalCausalLM.
  11. QLoRA configuration matches v0.2 controlled identity.
  12. Smoke command resolution: CLI parsing, data limiting to 12/2, stopping before GPU model setup.
  13. Checkpoint path isolation: verified under checkpoints/oloric-v0.3-smoke/.

Exit code: 0 = all checks passed, 1 = at least one failed.
"""
import copy
import json
import os
import subprocess
import sys
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
TRAIN_JSONL = os.path.join(REPO_ROOT, "data", "splits_v03", "train.jsonl")
VAL_JSONL = os.path.join(REPO_ROOT, "data", "splits_v03", "validation.jsonl")
DATASET_JSONL = os.path.join(REPO_ROOT, "data", "generated", "oloric_v03_dataset.jsonl")
SMOKE_DIR = os.path.join(REPO_ROOT, "checkpoints", "oloric-v0.3-smoke")
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
    print("=" * 64)
    print("OLORIC v0.3 SMOKE PREFLIGHT (CPU only — no model weights loaded)")
    print("=" * 64)

    # ------------------------------------------------------------------
    # 1. v0.3 train/validation loading
    # ------------------------------------------------------------------
    print("\n--- 1. v0.3 train/validation loading ---")
    try:
        sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
        from train_qlora import load_dataset as pipeline_load_dataset

        train_examples = pipeline_load_dataset(TRAIN_JSONL)
        val_examples = pipeline_load_dataset(VAL_JSONL)
        train_count = len(train_examples)
        val_count = len(val_examples)
        metrics["train_count"] = train_count
        metrics["val_count"] = val_count

        expected_counts = (train_count == 384 and val_count == 48)
        record(
            "V03_DATASET_LOAD",
            expected_counts,
            f"train={train_count} (expect 384), val={val_count} (expect 48)",
        )
        if not expected_counts:
            return finish()

        train_subset = [ex.dict() for ex in train_examples[:N_TRAIN]]
        val_subset = [ex.dict() for ex in val_examples[:N_VAL]]
        sample_ids = [d["id"] for d in train_subset]
        metrics["sample_ids"] = sample_ids
    except Exception as e:
        record("V03_DATASET_LOAD", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 2. v0.3 ID/dataset identity
    # ------------------------------------------------------------------
    print("\n--- 2. v0.3 ID/dataset identity ---")
    try:
        all_train_ids = [ex.id for ex in train_examples]
        all_val_ids = [ex.id for ex in val_examples]
        all_ids = all_train_ids + all_val_ids

        all_v03 = all(i.startswith("v03_") for i in all_ids)
        no_v02 = not any("v02" in i for i in all_ids)
        unique_ids = len(all_ids) == len(set(all_ids))

        record(
            "v0.3 ID identity (v03_ prefix, no v0.2 IDs, unique)",
            all_v03 and no_v02 and unique_ids,
            f"total_ids={len(all_ids)}, all_v03={all_v03}, no_v02={no_v02}, unique={unique_ids}",
        )
    except Exception as e:
        record("v0.3 ID identity", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 3. Strict content validator check on full v0.3 dataset
    # ------------------------------------------------------------------
    print("\n--- 3. Strict v0.3 content validation ---")
    try:
        from validate_dataset_content import validate_file

        passed_content, errs, warns, stats = validate_file(DATASET_JSONL)
        metrics["content_errors"] = len(errs)
        metrics["content_warnings"] = len(warns)
        record(
            "V03_CONTENT_VALIDATION",
            passed_content and len(errs) == 0,
            f"placeholder_errors={len(errs)}, review_warnings={len(warns)}, file={DATASET_JSONL}",
        )
        if not passed_content:
            for err in errs[:5]:
                print(f"  [ERR] {err}")
            return finish()
    except Exception as e:
        record("V03_CONTENT_VALIDATION", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 4. Schema validation of real v0.3 records
    # ------------------------------------------------------------------
    print("\n--- 4. Schema validation of real v0.3 records ---")
    try:
        schema_ok = True
        schema_err = ""
        for i, raw_rec in enumerate(train_subset):
            # Test TrainingExample validation
            te = TrainingExample(**raw_rec)
            # Test OloricModelInput validation
            mi = example_to_model_input(raw_rec)
            assert mi.learner_state.concept, "Missing concept in learner state"
            assert mi.document_context.title, "Missing title in document context"
            assert len(mi.conversation_context) > 0, "Empty conversation context"

        record(
            "V03_SCHEMA_VALIDATION",
            True,
            f"validated {len(train_subset)} sample records through TrainingExample and OloricModelInput schemas",
        )
    except Exception as e:
        record("V03_SCHEMA_VALIDATION", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 5. Tokenizer loading (Qwen/Qwen3-4B-Instruct-2507)
    # ------------------------------------------------------------------
    print("\n--- 5. Tokenizer initialization ---")
    try:
        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B-Instruct-2507",
            revision="cdbee75f17c01a7cc42f958dc650907174af0554",
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
    # 6. Real v0.3 prompt construction (task, instruction, context preservation)
    # ------------------------------------------------------------------
    print("\n--- 6. Real v0.3 prompt construction ---")
    try:
        test_rec = train_subset[0]
        model_input = example_to_model_input(test_rec)
        prompt_text = formatter.format_input(model_input)

        has_task = model_input.task in prompt_text
        has_instruction = model_input.instruction in prompt_text
        has_doc = (
            model_input.document_context.title in prompt_text
            and model_input.document_context.section in prompt_text
        )
        has_conv = all(
            turn.content in prompt_text
            for turn in model_input.conversation_context
        )

        record(
            "V03_TASK_PRESERVATION",
            has_task,
            f"task={model_input.task!r} embedded in system prompt",
        )
        record(
            "V03_INSTRUCTION_PRESERVATION",
            has_instruction,
            f"instruction={model_input.instruction!r} embedded in system prompt",
        )
        record(
            "v0.3 prompt context preservation (conversation + document)",
            has_doc and has_conv,
            f"doc_title={model_input.document_context.title!r}, turns={len(model_input.conversation_context)} preserved",
        )
    except Exception as e:
        record("V03_TASK_PRESERVATION", False, f"{type(e).__name__}: {e}")
        record("V03_INSTRUCTION_PRESERVATION", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 7. Differential prompt test
    # ------------------------------------------------------------------
    print("\n--- 7. Differential prompt test ---")
    try:
        rec_a = copy.deepcopy(train_subset[0])
        rec_b = copy.deepcopy(train_subset[0])

        rec_a["task"] = "task_alpha"
        rec_a["instruction"] = "Explain concept using simple words for children."
        rec_b["task"] = "task_beta"
        rec_b["instruction"] = "Provide practice exam questions with numerical calculations."

        # Identical conversation turns
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
        has_a = ("task_alpha" in p_a) and ("simple words" in p_a)
        has_b = ("task_beta" in p_b) and ("practice exam questions" in p_b)

        record(
            "V03_PROMPT_DIFFERENTIATION",
            different and has_a and has_b,
            "two records with identical conversation but different task/instruction produce distinct prompts",
        )
    except Exception as e:
        record("V03_PROMPT_DIFFERENTIATION", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 8. Tokenization & Sequence Length
    # ------------------------------------------------------------------
    print("\n--- 8. Tokenization test ---")
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
            "V03_TOKENIZATION",
            has_prompt and has_target and (seq_len == MAX_LENGTH),
            f"seq_len={MAX_LENGTH}, prompt+target present in decoded token stream",
        )
    except Exception as e:
        record("V03_TOKENIZATION", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 9. Label masking pipeline
    # ------------------------------------------------------------------
    print("\n--- 9. Label masking pipeline ---")
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
            "V03_LABEL_PIPELINE",
            n_active > 0 and n_masked > 0 and strict_labels and prompt_masked and pad_masked,
            f"active={n_active}, masked={n_masked}/{MAX_LENGTH}, prompt_masked={prompt_masked}, pad_masked={pad_masked}",
        )
    except Exception as e:
        record("V03_LABEL_PIPELINE", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 10. Collator & HF Trainer batch assembly
    # ------------------------------------------------------------------
    print("\n--- 10. Collator & HF Trainer batch assembly ---")
    try:
        ds = OloricTorchDataset(train_subset, formatter, max_length=MAX_LENGTH)
        batch = default_data_collator([ds[0], ds[1]])
        assert batch["input_ids"].shape == (2, MAX_LENGTH)
        assert batch["labels"].dtype == torch.long
        assert int((batch["labels"][0] != -100).sum()) > 0
        record(
            "V03_COLLATOR",
            True,
            f"default_data_collator preserves labels; batch input_ids shape={tuple(batch['input_ids'].shape)}",
        )
    except Exception as e:
        record("V03_COLLATOR", False, f"{type(e).__name__}: {e}")
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
            "V03_TRAINER_BATCH",
            batch_ok,
            f"pulled batch keys={sorted(pulled_batch.keys())}, active_targets_in_batch=True",
        )
    except Exception as e:
        record("V03_TRAINER_BATCH", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 11. QLoRA configuration check
    # ------------------------------------------------------------------
    print("\n--- 11. QLoRA configuration check ---")
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
            "V03_QLORA_CONFIG",
            ident,
            f"model={bm_cfg.get('name')}, NF4=True, double_quant=True, bf16=True, "
            f"lora(r={lora.get('r')}, a={lora.get('lora_alpha')}, do={lora.get('lora_dropout')}), "
            f"optim={resolved['optim']}, lr={resolved['learning_rate']}, ga={resolved['gradient_accumulation_steps']}",
        )
    except Exception as e:
        record("V03_QLORA_CONFIG", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 12. Checkpoint path isolation
    # ------------------------------------------------------------------
    print("\n--- 12. Checkpoint path isolation ---")
    try:
        final_dir = os.path.join(SMOKE_DIR, "final_model")
        inter_dir = os.path.join(SMOKE_DIR, "trainer_intermediate")
        os.makedirs(final_dir, exist_ok=True)
        os.makedirs(inter_dir, exist_ok=True)

        # Isolation checks: smoke dir is completely distinct from v0.2 dirs
        v02_dir = os.path.join(REPO_ROOT, "checkpoints", "oloric-v0.2")
        v02_smoke_dir = os.path.join(REPO_ROOT, "checkpoints", "oloric-v0.2-smoke")

        isolated = (
            os.path.abspath(SMOKE_DIR) != os.path.abspath(v02_dir)
            and os.path.abspath(SMOKE_DIR) != os.path.abspath(v02_smoke_dir)
            and "oloric-v0.3-smoke" in SMOKE_DIR
        )

        record(
            "V03_CHECKPOINT_ISOLATION",
            isolated and os.path.isdir(final_dir) and os.path.isdir(inter_dir),
            f"smoke_dir={SMOKE_DIR} (isolated from {v02_dir} and {v02_smoke_dir})",
        )
    except Exception as e:
        record("V03_CHECKPOINT_ISOLATION", False, f"{type(e).__name__}: {e}")
        return finish()

    # ------------------------------------------------------------------
    # 13. Smoke command resolution through CLI parser & loader
    # ------------------------------------------------------------------
    print("\n--- 13. Smoke command resolution ---")
    smoke_cmd = [
        sys.executable,
        "scripts/train_qlora.py",
        "--train-file",
        "data/splits_v03/train.jsonl",
        "--val-file",
        "data/splits_v03/validation.jsonl",
        "--output-dir",
        "checkpoints/oloric-v0.3-smoke",
        "--trainer-output-dir",
        "checkpoints/oloric-v0.3-smoke/trainer_intermediate",
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
            "V03_COMMAND_READY",
            cmd_ok,
            "smoke command parsed CLI args, loaded v0.3 data, applied 12/2 sample limits and overrides, "
            "and halted cleanly at GPU model setup boundary",
        )
    except Exception as e:
        record("V03_COMMAND_READY", False, f"{type(e).__name__}: {e}")
        return finish()

    return finish()


def finish() -> int:
    print("\n" + "=" * 64)
    print("PREFLIGHT SUMMARY")
    print("=" * 64)
    failed = 0
    for name, passed, _ in results:
        if not passed:
            failed += 1
    passed_count = len(results) - failed
    print(f"checks run: {len(results)}, passed: {passed_count}, failed: {failed}")
    all_passed = (failed == 0)
    print("OVERALL RESULT:", "PASS" if all_passed else "FAIL")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
