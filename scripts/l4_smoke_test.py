#!/usr/bin/env python3
"""
Gate 1 — Phase 3: L4 GPU Smoke Test
Self-contained QLoRA smoke test for NVIDIA L4 24GB.

Test A: Synthetic minimal (5 steps, seq_len 256)
Test B: Real Oloric examples at production config (10 steps, seq_len 2048)

Self-contained: no imports from oloric.src.*
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# VRAM measurement helpers
# ---------------------------------------------------------------------------

def get_vram_stats(device: int = 0) -> Dict[str, float]:
    """Get current VRAM statistics in GB."""
    import torch
    return {
        "allocated_gb": round(torch.cuda.memory_allocated(device) / (1024 ** 3), 3),
        "reserved_gb": round(torch.cuda.memory_reserved(device) / (1024 ** 3), 3),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved(device) / (1024 ** 3), 3),
    }


def get_hardware_info(device: int = 0) -> Dict[str, Any]:
    """Get hardware information."""
    import torch
    props = torch.cuda.get_device_properties(device)
    return {
        "gpu_name": torch.cuda.get_device_name(device),
        "total_vram_gb": round(props.total_mem / (1024 ** 3), 2),
        "cuda_version": torch.version.cuda or "N/A",
        "torch_version": torch.__version__,
        "gpu_count": torch.cuda.device_count(),
        "compute_capability": f"{props.major}.{props.minor}",
    }


# ---------------------------------------------------------------------------
# Synthetic data for Test A
# ---------------------------------------------------------------------------

SYNTHETIC_EXAMPLES = [
    {
        "instruction": "Explain the concept of supply and demand.",
        "input": "Student: I don't understand supply and demand.",
        "output": '{"action":"explain","strategy":"simple_explanation","difficulty":"beginner","response":"Supply and demand is the fundamental economic concept."}'
    },
    {
        "instruction": "Provide a concrete example of photosynthesis.",
        "input": "Student: What is photosynthesis?",
        "output": '{"action":"example","strategy":"concrete_example","difficulty":"beginner","response":"Photosynthesis is the process by which plants convert sunlight into energy."}'
    },
    {
        "instruction": "Address the misconception about gravity.",
        "input": "Student: I think heavier objects fall faster.",
        "output": '{"action":"misconception_correction","strategy":"misconception_correction","difficulty":"beginner","response":"Actually, in a vacuum, all objects fall at the same rate regardless of mass."}'
    },
    {
        "instruction": "Identify prerequisite knowledge for calculus.",
        "input": "Student: I want to learn calculus but I struggle with it.",
        "output": '{"action":"prerequisite","strategy":"prerequisite","difficulty":"intermediate","response":"Before calculus, you need a solid understanding of algebra and trigonometry."}'
    },
    {
        "instruction": "Generate follow-up questions about Newton laws.",
        "input": "Student: I think I understand Newton laws now.",
        "output": '{"action":"confirm_understanding","strategy":"follow_up_questions","difficulty":"beginner","response":"Great! Let me check your understanding with a few questions."}'
    },
]


# ---------------------------------------------------------------------------
# Load real Oloric examples for Test B
# ---------------------------------------------------------------------------

def load_real_examples(data_file: str, max_examples: int = 20) -> List[Dict[str, str]]:
    """Load real training examples and format them as instruction/input/output."""
    examples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Build instruction
            instruction = item.get("instruction", "Help the student.")

            # Build input from context
            ctx = item.get("context", {})
            doc = ctx.get("document_context", {})
            learner = ctx.get("learner_state", {})
            conv = ctx.get("conversation_context", [])

            input_parts = [
                f"Document: {doc.get('title', 'N/A')} — {doc.get('section', 'N/A')}",
                f"Selected text: {doc.get('selected_text', 'N/A')[:200]}",
                f"Student level: {learner.get('level', 'beginner')}",
                f"Concept: {learner.get('concept', 'N/A')}",
                f"Mastery: {learner.get('mastery', 0.0)}",
            ]
            for turn in conv[-3:]:  # last 3 turns
                role = turn.get("role", "student")
                content = turn.get("content", "")
                input_parts.append(f"{role}: {content}")

            input_text = "\n".join(input_parts)

            # Build output from target
            target = item.get("target", {})
            output_text = json.dumps(target, ensure_ascii=False)

            examples.append({
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
            })

            if len(examples) >= max_examples:
                break

    return examples


# ---------------------------------------------------------------------------
# Tokenize examples
# ---------------------------------------------------------------------------

def tokenize_examples(examples: List[Dict[str, str]], tokenizer, max_length: int):
    """Tokenize examples into a format suitable for training."""
    from torch.utils.data import Dataset

    class SimpleDataset(Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.encodings.items()}

    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for ex in examples:
        # Build the full text: instruction + input + output
        full_text = f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n{ex['output']}"

        # Tokenize
        encoded = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        # Mask padding tokens in labels
        labels[labels == tokenizer.pad_token_id] = -100

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

    import torch
    encodings = {
        "input_ids": torch.stack(all_input_ids),
        "attention_mask": torch.stack(all_attention_masks),
        "labels": torch.stack(all_labels),
    }

    return SimpleDataset(encodings)


# ---------------------------------------------------------------------------
# Run a single smoke test
# ---------------------------------------------------------------------------

def run_test(
    test_name: str,
    examples: List[Dict[str, str]],
    max_length: int,
    num_steps: int,
    gradient_accumulation: int,
    output_dir: str,
) -> Dict[str, Any]:
    """Run a single smoke test and return measurements."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

    results: Dict[str, Any] = {
        "test_name": test_name,
        "num_examples": len(examples),
        "max_length": max_length,
        "num_steps": num_steps,
        "gradient_accumulation": gradient_accumulation,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    checkpoint_dir = os.path.join(output_dir, f"checkpoint_{test_name}")

    try:
        # --- Step 1: Baseline VRAM ---
        torch.cuda.reset_peak_memory_stats(0)
        gc.collect()
        torch.cuda.empty_cache()
        results["vram_baseline"] = get_vram_stats()

        # --- Step 2: Load model in 4-bit ---
        print(f"\n  Loading model in 4-bit quantization...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
        results["model_load_time_seconds"] = round(time.time() - t0, 1)
        results["vram_after_model_load"] = get_vram_stats()
        print(f"  Model loaded. VRAM: {results['vram_after_model_load']['allocated_gb']:.2f} GB")

        # --- Step 3: Load tokenizer ---
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # --- Step 4: Prepare for kbit training ---
        model = prepare_model_for_kbit_training(model)
        results["vram_after_kbit_prep"] = get_vram_stats()

        # --- Step 5: Attach LoRA ---
        print(f"  Attaching QLoRA adapters...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        results["vram_after_qlora_attach"] = get_vram_stats()
        print(f"  QLoRA attached. VRAM: {results['vram_after_qlora_attach']['allocated_gb']:.2f} GB")

        # Print trainable parameters
        trainable, total = 0, 0
        for _, p in model.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        results["trainable_params"] = trainable
        results["total_params"] = total
        results["trainable_pct"] = round(100 * trainable / max(total, 1), 2)
        print(f"  Trainable: {trainable:,} / {total:,} ({results['trainable_pct']:.2f}%)")

        # --- Step 6: Tokenize data ---
        print(f"  Tokenizing {len(examples)} examples (max_length={max_length})...")
        dataset = tokenize_examples(examples, tokenizer, max_length)
        print(f"  Dataset ready: {len(dataset)} examples")

        # --- Step 7: Training ---
        print(f"  Starting training ({num_steps} steps)...")
        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            num_train_epochs=999,  # we'll stop by max_steps
            max_steps=num_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=gradient_accumulation,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            learning_rate=2e-4,
            weight_decay=0.01,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=1,
            save_steps=num_steps,  # save at end
            save_total_limit=1,
            fp16=False,
            bf16=True,
            tf32=True,
            dataloader_pin_memory=False,
            report_to="none",
            remove_unused_columns=False,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        torch.cuda.reset_peak_memory_stats(0)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        train_result = trainer.train()
        results["training_step_success"] = "PASS"
        results["training_loss"] = round(train_result.training_loss, 4)
        results["vram_after_training"] = get_vram_stats()

        # Check if loss is finite
        if train_result.training_loss != train_result.training_loss:  # NaN check
            results["training_step_success"] = "FAIL"
            results["training_failure_reason"] = "Loss is NaN"

        # Log history for loss trend
        log_history = trainer.state.log_history
        losses = [l.get("loss") for l in log_history if "loss" in l]
        results["loss_history"] = losses
        if len(losses) >= 2:
            results["loss_decreasing"] = "YES" if losses[-1] < losses[0] else "NO"
        else:
            results["loss_decreasing"] = "INCONCLUSIVE"

        print(f"  Training complete. Loss: {train_result.training_loss:.4f}")
        print(f"  Peak VRAM (allocated): {results['vram_after_training']['peak_allocated_gb']:.2f} GB")
        print(f"  Peak VRAM (reserved):  {results['vram_after_training']['peak_reserved_gb']:.2f} GB")

        # --- Step 8: Save checkpoint ---
        print(f"  Saving checkpoint...")
        try:
            trainer.save_model(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            results["checkpoint_save"] = "PASS"
            results["checkpoint_path"] = checkpoint_dir
            print(f"  Checkpoint saved: {checkpoint_dir}")
        except Exception as e:
            results["checkpoint_save"] = "FAIL"
            results["checkpoint_save_error"] = str(e)
            print(f"  Checkpoint save FAILED: {e}")

        # --- Step 9: Free model, reload from checkpoint ---
        print(f"  Freeing model for reload test...")
        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()

        print(f"  Reloading base model + checkpoint...")
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
            )
            reloaded_model = PeftModel.from_pretrained(base_model, checkpoint_dir)
            reloaded_model.eval()
            results["checkpoint_reload"] = "PASS"
            print(f"  Checkpoint reloaded successfully")
        except Exception as e:
            results["checkpoint_reload"] = "FAIL"
            results["checkpoint_reload_error"] = str(e)
            print(f"  Checkpoint reload FAILED: {e}")
            return results

        # --- Step 10: Inference after reload ---
        print(f"  Running inference on reloaded model...")
        try:
            test_prompt = "### Instruction:\nExplain gravity simply.\n\n### Input:\nStudent: What is gravity?\n\n### Response:\n"
            inputs = tokenizer(
                test_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(reloaded_model.device)

            with torch.no_grad():
                outputs = reloaded_model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            results["inference_after_reload"] = "PASS"
            results["inference_output_sample"] = generated[:200]
            print(f"  Inference output: {generated[:120]}...")
        except Exception as e:
            results["inference_after_reload"] = "FAIL"
            results["inference_error"] = str(e)
            print(f"  Inference FAILED: {e}")

        # Cleanup
        del reloaded_model
        del base_model
        gc.collect()
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError as e:
        results["training_step_success"] = "FAIL"
        results["oom_error"] = str(e)
        results["vram_at_oom"] = get_vram_stats()
        print(f"  OOM ERROR: {e}")

    except Exception as e:
        results["training_step_success"] = "FAIL"
        results["error"] = str(e)
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    return results


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def run_dry(data_file: str, output_dir: str, tests: List[str]):
    """Validate script logic without CUDA."""
    print("=" * 60)
    print("DRY RUN — validating smoke test logic (no GPU required)")
    print("=" * 60)

    print(f"Tests to run: {tests}")
    print(f"Output dir: {output_dir}")

    if "real" in tests or "both" in tests:
        if not os.path.exists(data_file):
            print(f"WARNING: data file not found: {data_file}")
        else:
            examples = load_real_examples(data_file, max_examples=20)
            print(f"Loaded {len(examples)} real examples")
            if examples:
                print(f"  First example instruction: {examples[0]['instruction'][:80]}")
                print(f"  First example input length: {len(examples[0]['input'])} chars")
                print(f"  First example output length: {len(examples[0]['output'])} chars")

    print(f"\nSynthetic examples: {len(SYNTHETIC_EXAMPLES)}")
    for i, ex in enumerate(SYNTHETIC_EXAMPLES):
        print(f"  [{i+1}] {ex['instruction'][:60]}")

    os.makedirs(output_dir, exist_ok=True)

    # Write dry-run report
    dry_report = {
        "dry_run": True,
        "tests": tests,
        "synthetic_examples": len(SYNTHETIC_EXAMPLES),
        "data_file": data_file,
        "data_file_exists": os.path.exists(data_file),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    report_path = os.path.join(output_dir, "dryrun_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(dry_report, f, indent=2)
    print(f"\nDry-run report saved: {report_path}")

    print("\nDRY RUN PASSED — smoke test logic is valid")
    return 0


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_smoke_tests(data_file: str, output_dir: str, tests: List[str]):
    """Run actual smoke tests on GPU."""
    import torch
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected. Use --dry-run for CPU validation.")
        return 1

    hw = get_hardware_info()
    print("=" * 60)
    print("L4 GPU SMOKE TEST")
    print("=" * 60)
    print(f"GPU: {hw['gpu_name']}")
    print(f"Total VRAM: {hw['total_vram_gb']} GB")
    print(f"CUDA: {hw['cuda_version']}")
    print(f"PyTorch: {hw['torch_version']}")

    os.makedirs(output_dir, exist_ok=True)

    # Save hardware info
    hw_path = os.path.join(output_dir, "hardware.json")
    with open(hw_path, "w", encoding="utf-8") as f:
        json.dump(hw, f, indent=2)

    # Get library versions
    import transformers, peft, bitsandbytes
    hw["transformers_version"] = transformers.__version__
    hw["peft_version"] = peft.__version__
    hw["bitsandbytes_version"] = bitsandbytes.__version__
    with open(hw_path, "w", encoding="utf-8") as f:
        json.dump(hw, f, indent=2)

    all_measurements = {"hardware": hw, "tests": {}}

    # Test A: Synthetic minimal
    if "synthetic" in tests or "both" in tests:
        print("\n" + "=" * 60)
        print("TEST A: Synthetic Minimal")
        print("  Examples: 5 synthetic, Steps: 5, SeqLen: 256")
        print("=" * 60)

        results_a = run_test(
            test_name="synthetic",
            examples=SYNTHETIC_EXAMPLES,
            max_length=256,
            num_steps=5,
            gradient_accumulation=1,
            output_dir=output_dir,
        )
        all_measurements["tests"]["synthetic"] = results_a

    # Test B: Real Oloric examples
    if "real" in tests or "both" in tests:
        print("\n" + "=" * 60)
        print("TEST B: Real Oloric Examples")
        print("  Examples: 10-20 real, Steps: 10, SeqLen: 2048")
        print("=" * 60)

        if not os.path.exists(data_file):
            print(f"ERROR: data file not found: {data_file}")
            all_measurements["tests"]["real"] = {"error": f"File not found: {data_file}"}
        else:
            real_examples = load_real_examples(data_file, max_examples=20)
            if len(real_examples) < 10:
                print(f"WARNING: only {len(real_examples)} examples loaded (wanted 10-20)")

            results_b = run_test(
                test_name="real",
                examples=real_examples,
                max_length=2048,
                num_steps=10,
                gradient_accumulation=4,
                output_dir=output_dir,
            )
            all_measurements["tests"]["real"] = results_b

    # Save all measurements
    meas_path = os.path.join(output_dir, "measurements.json")
    with open(meas_path, "w", encoding="utf-8") as f:
        json.dump(all_measurements, f, indent=2, default=str)
    print(f"\nMeasurements saved: {meas_path}")

    # Generate report
    report_path = os.path.join(output_dir, "report.md")
    generate_smoke_report(all_measurements, report_path)
    print(f"Report saved: {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    for test_name, results in all_measurements["tests"].items():
        if isinstance(results, dict) and "error" not in results:
            step_ok = results.get("training_step_success", "UNKNOWN")
            ckpt_save = results.get("checkpoint_save", "UNKNOWN")
            ckpt_reload = results.get("checkpoint_reload", "UNKNOWN")
            infer_ok = results.get("inference_after_reload", "UNKNOWN")
            peak = results.get("vram_after_training", {}).get("peak_allocated_gb", "?")
            print(f"  {test_name}: step={step_ok} ckpt_save={ckpt_save} "
                  f"ckpt_reload={ckpt_reload} inference={infer_ok} peak_vram={peak}GB")
        else:
            print(f"  {test_name}: ERROR — {results.get('error', 'unknown')}")

    return 0


def generate_smoke_report(measurements: Dict[str, Any], report_path: str):
    """Generate a markdown report from smoke test measurements."""
    hw = measurements.get("hardware", {})
    lines = [
        "# L4 GPU Smoke Test Report",
        "",
        f"**Generated**: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Hardware",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| GPU | {hw.get('gpu_name', 'N/A')} |",
        f"| Total VRAM | {hw.get('total_vram_gb', 'N/A')} GB |",
        f"| CUDA | {hw.get('cuda_version', 'N/A')} |",
        f"| PyTorch | {hw.get('torch_version', 'N/A')} |",
        f"| Transformers | {hw.get('transformers_version', 'N/A')} |",
        f"| PEFT | {hw.get('peft_version', 'N/A')} |",
        f"| bitsandbytes | {hw.get('bitsandbytes_version', 'N/A')} |",
        "",
    ]

    for test_name, results in measurements.get("tests", {}).items():
        lines.append(f"## Test: {test_name}")
        lines.append("")

        if "error" in results:
            lines.append(f"**ERROR**: {results['error']}")
            lines.append("")
            continue

        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Examples | {results.get('num_examples', 'N/A')} |")
        lines.append(f"| Max Length | {results.get('max_length', 'N/A')} |")
        lines.append(f"| Steps | {results.get('num_steps', 'N/A')} |")
        lines.append(f"| Grad Accum | {results.get('gradient_accumulation', 'N/A')} |")
        lines.append(f"| Trainable Params | {results.get('trainable_params', 'N/A'):,} ({results.get('trainable_pct', 'N/A')}%) |")

        # VRAM stages
        for stage_key, stage_label in [
            ("vram_after_model_load", "After Model Load"),
            ("vram_after_kbit_prep", "After kbit Prep"),
            ("vram_after_qlora_attach", "After QLoRA Attach"),
            ("vram_after_training", "After Training (Peak)"),
        ]:
            vram = results.get(stage_key, {})
            if vram:
                lines.append(f"| VRAM {stage_label} (alloc) | {vram.get('allocated_gb', 'N/A')} GB |")
                lines.append(f"| VRAM {stage_label} (reserved) | {vram.get('reserved_gb', 'N/A')} GB |")
                if "peak_allocated_gb" in vram:
                    lines.append(f"| VRAM {stage_label} (peak alloc) | {vram.get('peak_allocated_gb', 'N/A')} GB |")
                    lines.append(f"| VRAM {stage_label} (peak reserved) | {vram.get('peak_reserved_gb', 'N/A')} GB |")

        lines.append(f"| Training Loss | {results.get('training_loss', 'N/A')} |")
        lines.append(f"| Loss Decreasing | {results.get('loss_decreasing', 'N/A')} |")
        lines.append(f"| Training Step | {results.get('training_step_success', 'N/A')} |")
        lines.append(f"| Checkpoint Save | {results.get('checkpoint_save', 'N/A')} |")
        lines.append(f"| Checkpoint Reload | {results.get('checkpoint_reload', 'N/A')} |")
        lines.append(f"| Inference After Reload | {results.get('inference_after_reload', 'N/A')} |")

        if results.get("inference_output_sample"):
            lines.append("")
            lines.append("### Inference Output Sample")
            lines.append(f"```\n{results['inference_output_sample']}\n```")

        if results.get("loss_history"):
            lines.append("")
            lines.append("### Loss History")
            for step, loss in enumerate(results["loss_history"]):
                lines.append(f"- Step {step + 1}: {loss}")

        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gate 1 — L4 GPU Smoke Test"
    )
    parser.add_argument(
        "--test", type=str, default="both",
        choices=["synthetic", "real", "both"],
        help="Which test(s) to run",
    )
    parser.add_argument(
        "--data-file", type=str,
        default="data/generated/enhanced_seed_data.jsonl",
        help="Path to real Oloric training data (for Test B)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="evaluation/smoke_test",
        help="Output directory for measurements and reports",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate script logic without GPU",
    )
    args = parser.parse_args()

    tests = ["synthetic", "real"] if args.test == "both" else [args.test]

    if args.dry_run:
        return run_dry(args.data_file, args.output_dir, tests)
    else:
        return run_smoke_tests(args.data_file, args.output_dir, tests)


if __name__ == "__main__":
    sys.exit(main())
