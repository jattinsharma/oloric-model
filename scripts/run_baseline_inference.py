#!/usr/bin/env python3
"""
Gate 1 — Phase 1: Actual Baseline Inference
Runs Qwen/Qwen3-4B-Instruct-2507 (unfine-tuned) on benchmark scenarios.

Two modes:
  A) general  — generic tutoring prompt, free-form output
  B) structured — full Oloric system message with JSON contract

Self-contained: no imports from oloric.src.* (those are broken).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_general_system_prompt(item: Dict[str, Any]) -> str:
    """Mode A: generic tutoring instruction — no Oloric schema."""
    ctx = item["context"]
    doc = ctx["document_context"]
    learner = ctx["learner_state"]

    lines = [
        "You are an expert tutor. Help the student understand the concept they are struggling with.",
        "Adapt your explanation to the student's level and prior knowledge.",
        "",
        f"Subject: {item['domain']}",
        f"Document: {doc['title']} — Section: {doc['section']} (Page {doc['page']})",
        f"Selected text: \"{doc['selected_text']}\"",
        f"Surrounding context: {doc['surrounding_context'][:300]}",
    ]
    if doc.get("retrieved_evidence"):
        lines.append("Relevant evidence:")
        for ev in doc["retrieved_evidence"][:3]:
            lines.append(f"  - {ev}")
    lines += [
        "",
        f"Student level: {learner['level']}",
        f"Concept: {learner['concept']}",
        f"Mastery: {learner['mastery']:.0%}",
        f"Known prerequisites: {', '.join(learner.get('known_prerequisites', [])) or 'none'}",
        f"Weak prerequisites: {', '.join(learner.get('weak_prerequisites', [])) or 'none'}",
        f"Known misconceptions: {', '.join(learner.get('known_misconceptions', [])) or 'none'}",
        "",
        f"Current goal: {ctx['current_goal']}",
        "",
        f"Task: {item['instruction']}",
    ]
    return "\n".join(lines)


def build_structured_system_prompt(item: Dict[str, Any]) -> str:
    """Mode B: full Oloric system message with JSON output contract."""
    ctx = item["context"]
    doc = ctx["document_context"]
    learner = ctx["learner_state"]

    lines = [
        "You are OLORIC, an adaptive teaching model.",
        "Your goal is to help the student understand the concept by providing",
        "appropriate explanations and adapting your teaching strategy based on",
        "their responses.",
        "",
        "You MUST respond with a single JSON object matching this schema exactly:",
        "",
        "{",
        '  "action": "<one of: diagnose, explain, simplify, analogy, example,',
        '             prerequisite, misconception_correction, hint, practice,',
        '             recap, confirm_understanding, memory_candidate>",',
        '  "strategy": "<string describing the strategy used>",',
        '  "difficulty": "<beginner | intermediate | advanced>",',
        '  "response": "<your main tutoring response text>",',
        '  "understanding_check": {',
        '    "required": <true|false>,',
        '    "question": "<question to check understanding or null>",',
        '    "expected_answer": "<expected answer or null>"',
        "  },",
        '  "diagnosis": {',
        '    "confusion_type": "<conceptual | procedural | prerequisite_gap | misconception>",',
        '    "severity": "<low | medium | high>",',
        '    "misconception_addressed": "<string or null>"',
        "  },",
        '  "memory": {',
        '    "candidate": <true|false>,',
        '    "title": "<memory title or null>",',
        '    "content": "<memory content or null>",',
        '    "anchor_concept": "<concept this memory is anchored to or null>",',
        '    "memory_type": "<definition | procedure | example | misconception_correction | hint | prerequisite | null>",',
        '    "confidence": <0.0 to 1.0>',
        "  }",
        "}",
        "",
        "Document Context:",
        f"  Document: {doc['title']} (Page {doc['page']}, Section: {doc['section']})",
        f"  Selected Text: \"{doc['selected_text']}\"",
        f"  Surrounding Context: {doc['surrounding_context'][:300]}",
    ]
    if doc.get("retrieved_evidence"):
        lines.append("  Retrieved Evidence:")
        for ev in doc["retrieved_evidence"][:3]:
            lines.append(f"    - {ev}")
    lines += [
        "",
        "Learner State:",
        f"  Level: {learner['level']}",
        f"  Concept: {learner['concept']}",
        f"  Mastery: {learner['mastery']:.2f}",
        f"  Known Prerequisites: {', '.join(learner.get('known_prerequisites', [])) or 'None'}",
        f"  Weak Prerequisites: {', '.join(learner.get('weak_prerequisites', [])) or 'None'}",
        f"  Known Misconceptions: {', '.join(learner.get('known_misconceptions', [])) or 'None'}",
        "",
        f"Current Goal: {ctx['current_goal']}",
        "",
        f"Task: {item['instruction']}",
        "",
        "Respond with ONLY the JSON object. No markdown fences. No explanation outside the JSON.",
    ]
    return "\n".join(lines)


def build_messages(item: Dict[str, Any], mode: str) -> List[Dict[str, str]]:
    """Build the chat messages list for the model."""
    if mode == "general":
        system_prompt = build_general_system_prompt(item)
    else:
        system_prompt = build_structured_system_prompt(item)

    messages = [{"role": "system", "content": system_prompt}]

    conv = item["context"].get("conversation_context", [])
    for turn in conv:
        role = "user" if turn["role"] == "student" else "assistant"
        messages.append({"role": role, "content": turn["content"]})

    # If the last message is from assistant (tutor), add a user nudge
    if not conv or conv[-1]["role"] != "student":
        messages.append({"role": "user", "content": "Please help me understand this concept."})

    return messages


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_benchmark(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: invalid JSON on line {line_num}: {e}")
    return items


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------

def run_dry(benchmark_path: str, output_dir: str, modes: List[str]):
    """Validate script logic without loading the model."""
    print("=" * 60)
    print("DRY RUN — validating script logic (no model loaded)")
    print("=" * 60)

    items = load_benchmark(benchmark_path)
    print(f"Loaded {len(items)} benchmark items")
    print(f"Benchmark SHA-256: {sha256_file(benchmark_path)}")
    print(f"Output directory: {output_dir}")
    print(f"Modes: {modes}")

    for mode in modes:
        print(f"\n--- Mode: {mode} ---")
        for i, item in enumerate(items):
            messages = build_messages(item, mode)
            total_chars = sum(len(m["content"]) for m in messages)
            print(f"  [{i+1:02d}] {item['id'][:50]:50s}  msgs={len(messages)}  chars={total_chars}")

    # Validate output directory creation
    ensure_dir(output_dir)
    print(f"\nOutput directory created: {output_dir}")

    # Write a sample config
    config = {
        "dry_run": True,
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "benchmark_hash": sha256_file(benchmark_path),
        "benchmark_items": len(items),
        "modes": modes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    config_path = os.path.join(output_dir, "config_dryrun.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Dry-run config saved: {config_path}")

    print("\nDRY RUN PASSED — script logic is valid")
    return 0


# ---------------------------------------------------------------------------
# Real inference
# ---------------------------------------------------------------------------

def run_inference(benchmark_path: str, output_dir: str, model_name: str, modes: List[str]):
    """Run actual model inference on benchmark items."""
    print("=" * 60)
    print("ACTUAL BASELINE INFERENCE")
    print("=" * 60)

    # Check for CUDA
    import torch
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected. Cannot run actual inference.")
        print("Use --dry-run to validate script logic on CPU.")
        return 1

    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    total_mem_bytes = getattr(props, "total_memory", getattr(props, "total_mem", 0))
    total_vram = total_mem_bytes / (1024 ** 3)
    print(f"GPU: {gpu_name}")
    print(f"Total VRAM: {total_vram:.2f} GB")

    # Import heavy dependencies only when actually running
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Load benchmark
    items = load_benchmark(benchmark_path)
    print(f"Loaded {len(items)} benchmark items")

    # Prepare output directory
    ensure_dir(output_dir)

    # Quantization config (same as training — 4-bit NF4)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Generation config
    gen_kwargs = {
        "max_new_tokens": 1024,
        "temperature": 0.3,
        "top_p": 0.9,
        "do_sample": True,
    }

    # Load model
    print(f"\nLoading model: {model_name}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    model_load_time = time.time() - t0
    print(f"Model loaded in {model_load_time:.1f}s")

    model_load_vram = torch.cuda.memory_allocated(0) / (1024 ** 3)
    print(f"Model load VRAM (allocated): {model_load_vram:.2f} GB")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get library versions
    import transformers
    import peft
    import bitsandbytes
    lib_versions = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "cuda": torch.version.cuda or "N/A",
    }

    # Save config
    run_config = {
        "model_id": model_name,
        "library_versions": lib_versions,
        "generation_config": gen_kwargs,
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "bfloat16",
        },
        "gpu": {
            "name": gpu_name,
            "total_vram_gb": round(total_vram, 2),
            "model_load_vram_gb": round(model_load_vram, 2),
        },
        "benchmark_file": benchmark_path,
        "benchmark_hash": sha256_file(benchmark_path),
        "benchmark_items": len(items),
        "modes": modes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_load_time_seconds": round(model_load_time, 1),
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    print(f"Config saved: {config_path}")

    # Open output files
    prompts_path = os.path.join(output_dir, "prompts.jsonl")
    responses_path = os.path.join(output_dir, "responses.jsonl")
    prompts_f = open(prompts_path, "w", encoding="utf-8")
    responses_f = open(responses_path, "w", encoding="utf-8")

    total_runs = len(items) * len(modes)
    run_idx = 0

    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"MODE: {mode.upper()}")
        print(f"{'=' * 60}")

        for i, item in enumerate(items):
            run_idx += 1
            item_id = item["id"]
            print(f"\n[{run_idx}/{total_runs}] {item_id} ({mode})")

            # Build messages
            messages = build_messages(item, mode)

            # Format prompt using tokenizer chat template
            try:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                print(f"  WARNING: chat template failed ({e}), using fallback")
                parts = []
                for m in messages:
                    parts.append(f"<|{m['role']}|>\n{m['content']}")
                parts.append("<|assistant|>")
                formatted_prompt = "\n".join(parts)

            # Save prompt
            prompt_record = {
                "id": item_id,
                "mode": mode,
                "domain": item.get("domain", "unknown"),
                "category": item.get("category", "unknown"),
                "system_prompt": messages[0]["content"],
                "user_messages": [m for m in messages[1:]],
                "formatted_prompt_length": len(formatted_prompt),
            }
            prompts_f.write(json.dumps(prompt_record, ensure_ascii=False) + "\n")

            # Tokenize
            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(model.device)
            input_len = inputs.input_ids.shape[1]

            # Generate
            torch.cuda.reset_peak_memory_stats(0)
            t_start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **gen_kwargs,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen_time = time.time() - t_start

            # Decode only the generated portion
            output_ids = outputs[0][input_len:]
            raw_response = tokenizer.decode(output_ids, skip_special_tokens=True)

            peak_alloc = torch.cuda.max_memory_allocated(0) / (1024 ** 3)

            print(f"  input_tokens={input_len}  output_tokens={len(output_ids)}  "
                  f"time={gen_time:.1f}s  peak_vram={peak_alloc:.2f}GB")
            print(f"  response_preview: {raw_response[:120]}...")

            # Save response
            response_record = {
                "id": item_id,
                "mode": mode,
                "domain": item.get("domain", "unknown"),
                "category": item.get("category", "unknown"),
                "raw_response": raw_response,
                "generation_time_seconds": round(gen_time, 2),
                "input_token_count": input_len,
                "output_token_count": len(output_ids),
                "peak_vram_gb": round(peak_alloc, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            responses_f.write(json.dumps(response_record, ensure_ascii=False) + "\n")
            responses_f.flush()

    prompts_f.close()
    responses_f.close()

    print(f"\n{'=' * 60}")
    print(f"BASELINE INFERENCE COMPLETE")
    print(f"{'=' * 60}")
    print(f"Prompts saved:   {prompts_path}")
    print(f"Responses saved: {responses_path}")
    print(f"Config saved:    {config_path}")
    print(f"Total runs: {run_idx}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gate 1 — Baseline inference on Qwen3-4B-Instruct-2507"
    )
    parser.add_argument(
        "--benchmark", type=str,
        default="evaluation/benchmark.jsonl",
        help="Path to benchmark JSONL file",
    )
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["general", "structured", "both"],
        help="Baseline mode: general, structured, or both",
    )
    parser.add_argument(
        "--model", type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Model identifier on Hugging Face",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="evaluation/baseline",
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate script logic without loading model (CPU OK)",
    )
    args = parser.parse_args()

    modes = ["general", "structured"] if args.mode == "both" else [args.mode]

    if args.dry_run:
        return run_dry(args.benchmark, args.output_dir, modes)
    else:
        return run_inference(args.benchmark, args.output_dir, args.model, modes)


if __name__ == "__main__":
    sys.exit(main())
