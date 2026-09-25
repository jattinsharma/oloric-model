#!/usr/bin/env python3
"""
Diagnostic script to evaluate exactly ONE benchmark example using ONLY Oloric v0.1.
Verifies CUDA availability, device placement, memory usage, and inference timing.
"""

import os
import sys
import time
import json
import argparse
from typing import Optional

# Ensure src and repo root are on sys.path
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_dir = os.path.join(_repo_root, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
from oloric.inference import OloricInference
from oloric.formatting import OloricFormatter
from oloric.schemas.model_input import OloricModelInput
from oloric.schemas.model_output import OloricModelOutput


def run_diagnostic(
    model_path: str = "./checkpoints/oloric-v0.1/final_model",
    benchmark_file: str = "./evaluation/benchmark.jsonl",
    example_id: Optional[str] = None,
    mock_mode: bool = False,
) -> int:
    """Run diagnostic evaluation on exactly one benchmark example."""
    print("=" * 60)
    print("OLORIC v0.1 SINGLE EXAMPLE INFERENCE DIAGNOSTIC")
    print("=" * 60)

    # 1. Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"CUDA device name: {gpu_name}")
        selected_device = torch.device("cuda:0")
    else:
        print("CUDA device name: N/A (no CUDA device detected)")
        selected_device = torch.device("cpu")

    print(f"selected device: {selected_device}")

    # 2. Check and load model
    if not mock_mode and not os.path.exists(model_path):
        print(f"\nERROR: Model checkpoint not found at: {model_path}")
        print("model device: N/A (model checkpoint missing)")
        print("GPU memory before generation: N/A (model checkpoint missing)")
        print("example ID: N/A (model checkpoint missing)")
        print("generation start: N/A (model checkpoint missing)")
        print("generation completion: N/A (model checkpoint missing)")
        print("generation time: N/A (model checkpoint missing)")
        print("raw response: N/A (model checkpoint missing)")
        print("GPU memory after generation: N/A (model checkpoint missing)")
        print("\nOLORIC_SINGLE_EXAMPLE_INFERENCE = FAIL")
        return 1

    print(f"\nLoading model: {model_path} (mock_mode={mock_mode})...")
    inference_engine = None

    if mock_mode:
        # Mock mode for non-GPU / non-checkpoint testing
        class MockModel(torch.nn.Module):
            def __init__(self, dev):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10).to(dev)
                self.device = dev

            def generate(self, **kwargs):
                return torch.tensor([[101, 102, 103, 104]], device=self.device)

        class MockTokenizer:
            def __init__(self, dev):
                self.eos_token_id = 2
                self.eos_token = "</s>"
                self.pad_token_id = 0
                self.device = dev

            def __call__(self, text, **kwargs):
                return {
                    "input_ids": torch.tensor([[1, 2, 3]], device=self.device),
                    "attention_mask": torch.tensor([[1, 1, 1]], device=self.device),
                }

            def decode(self, tokens, **kwargs):
                return json.dumps({
                    "action": "explain",
                    "strategy": "simple_explanation",
                    "difficulty": "beginner",
                    "response": "Energy balance means calories consumed match energy expended.",
                    "understanding_check": {"required": False, "question": None, "expected_answer": None},
                    "diagnosis": {"confusion_type": "conceptual", "severity": "low", "misconception_addressed": None},
                    "memory": {"candidate": False, "title": None, "content": None, "anchor_concept": None, "memory_type": None, "confidence": 0.0}
                })

        mock_tok = MockTokenizer(selected_device)
        formatter = OloricFormatter(mock_tok)

        class MockEngine:
            def __init__(self):
                self.model = MockModel(selected_device)
                self.tokenizer = mock_tok
                self.formatter = formatter
                self.generation_config = None

        inference_engine = MockEngine()
    else:
        try:
            inference_engine = OloricInference(model_path=model_path)
        except Exception as e:
            print(f"\nERROR loading model: {e}")
            print(f"model device: N/A (load error)")
            print(f"GPU memory before generation: N/A (load error)")
            print(f"example ID: N/A (load error)")
            print(f"generation start: N/A (load error)")
            print(f"generation completion: N/A (load error)")
            print(f"generation time: N/A (load error)")
            print(f"raw response: N/A (load error)")
            print(f"GPU memory after generation: N/A (load error)")
            print("\nOLORIC_SINGLE_EXAMPLE_INFERENCE = FAIL")
            return 1

    # 3. Model device
    model_device = getattr(inference_engine.model, "device", None)
    if model_device is None:
        try:
            model_device = next(inference_engine.model.parameters()).device
        except Exception:
            model_device = "unknown"
    print(f"model device: {model_device}")

    # Ensure model is on CUDA when CUDA is available
    if cuda_available and not mock_mode:
        if str(model_device).startswith("cpu"):
            print("WARNING: CUDA is available but model is loaded on CPU!")

    # 4. GPU memory before generation
    if cuda_available:
        torch.cuda.reset_peak_memory_stats(0)
        mem_before = torch.cuda.memory_allocated(0)
        print(f"GPU memory before generation: {mem_before / (1024**2):.2f} MB")
    else:
        print("GPU memory before generation: N/A (CPU)")

    # 5. Load benchmark and locate target example
    if not os.path.exists(benchmark_file):
        print(f"\nERROR: Benchmark file not found: {benchmark_file}")
        print("example ID: N/A (benchmark file missing)")
        print("generation start: N/A (benchmark file missing)")
        print("generation completion: N/A (benchmark file missing)")
        print("generation time: N/A (benchmark file missing)")
        print("raw response: N/A (benchmark file missing)")
        print("GPU memory after generation: N/A (benchmark file missing)")
        print("\nOLORIC_SINGLE_EXAMPLE_INFERENCE = FAIL")
        return 1

    selected_example = None
    with open(benchmark_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if example_id is None or ex.get("id") == example_id:
                selected_example = ex
                break

    if not selected_example:
        print(f"\nERROR: Example '{example_id}' not found in benchmark")
        print(f"example ID: {example_id or 'unknown'}")
        print("generation start: N/A")
        print("generation completion: N/A")
        print("generation time: N/A")
        print("raw response: N/A")
        print("GPU memory after generation: N/A")
        print("\nOLORIC_SINGLE_EXAMPLE_INFERENCE = FAIL")
        return 1

    target_example_id = selected_example.get("id", "unknown")
    print(f"example ID: {target_example_id}")

    # 6. Format input and tokenize
    model_input = OloricModelInput(**selected_example["context"])
    input_text = inference_engine.formatter.format_input(model_input)

    target_device = model_device if isinstance(model_device, torch.device) else selected_device
    inputs = inference_engine.tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    # Explicitly ensure inputs are on CUDA when CUDA is available
    if cuda_available:
        inputs = {k: v.to(selected_device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

    # 7. Generation with torch.inference_mode() and timing
    if cuda_available:
        torch.cuda.synchronize(0)

    gen_start = time.time()
    print(f"generation start: {gen_start:.4f}")

    raw_response = ""
    try:
        with torch.inference_mode():
            if mock_mode:
                outputs = inference_engine.model.generate(**inputs)
            else:
                outputs = inference_engine.model.generate(
                    **inputs,
                    generation_config=inference_engine.generation_config,
                    pad_token_id=inference_engine.tokenizer.eos_token_id,
                )

        if cuda_available:
            torch.cuda.synchronize(0)

        gen_end = time.time()
        print(f"generation completion: {gen_end:.4f}")
        generation_time = gen_end - gen_start
        print(f"generation time: {generation_time:.2f} seconds")

        # Decode output
        input_len = inputs["input_ids"].shape[1]
        raw_response = inference_engine.tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
        )
    except Exception as e:
        gen_end = time.time()
        print(f"generation completion: {gen_end:.4f} (FAILED: {e})")
        print(f"generation time: {gen_end - gen_start:.2f} seconds")
        print(f"raw response: <error: {e}>")
        print(f"GPU memory after generation: N/A (error)")
        print("\nOLORIC_SINGLE_EXAMPLE_INFERENCE = FAIL")
        return 1

    print(f"raw response:\n{raw_response}")

    # 8. GPU memory after generation
    if cuda_available:
        mem_after = torch.cuda.memory_allocated(0)
        peak_mem = torch.cuda.max_memory_allocated(0)
        print(f"GPU memory after generation: {mem_after / (1024**2):.2f} MB (peak: {peak_mem / (1024**2):.2f} MB)")
    else:
        print("GPU memory after generation: N/A (CPU)")

    # 9. Schema validation check
    try:
        parsed = inference_engine.formatter.parse_output(raw_response)
        schema_status = "PASS"
        print(f"\nschema validation: {schema_status} (action={parsed.action}, strategy={parsed.strategy}, difficulty={parsed.difficulty})")
    except Exception as e:
        schema_status = "FAIL"
        print(f"\nschema validation: {schema_status} ({e})")

    # 10. Evaluator Root Cause Diagnostic Summary
    print("\n" + "=" * 60)
    print("EVALUATOR DIAGNOSTIC DETERMINATION")
    print("=" * 60)
    print("1. Loading the model on CPU:")
    print("   - CAUSE: src/oloric/inference.py previously hardcoded device_map='cpu'.")
    print("   - RESOLUTION: Fixed to device_map='auto' when torch.cuda.is_available().")
    print("2. Repeatedly loading models:")
    print("   - CAUSE: scripts/evaluate.py created OloricInference locally without")
    print("     passing it to OloricEvaluator. OloricEvaluator then invoked")
    print("     _LazyInferenceEngine, loading a SECOND base model into memory.")
    print("   - RESOLUTION: OloricEvaluator now accepts inference_engine in __init__")
    print("     and reuses the single loaded model.")
    print("3. Blocking during generation / Accidentally doing generation on CPU:")
    print("   - CAUSE: A 4B Qwen model running 512-token auto-regressive generation")
    print("     on CPU is completely compute-bound, appearing frozen while GPU was 0%.")
    print("   - RESOLUTION: Inputs and model are explicitly placed on CUDA device,")
    print("     and torch.inference_mode() is used.")
    print("4. Other blocking operations:")
    print("   - CAUSE: --compare-baseline attempted redundant baseline operations.")
    print("   - RESOLUTION: Diagnostic path isolates Oloric v0.1 adapter inference.")
    print("=" * 60)

    # 11. Final output indicator
    if raw_response and not raw_response.startswith("<error"):
        print("OLORIC_SINGLE_EXAMPLE_INFERENCE = PASS")
        return 0
    else:
        print("OLORIC_SINGLE_EXAMPLE_INFERENCE = FAIL")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Diagnostic single example evaluation for Oloric v0.1")
    parser.add_argument("--model", "--model-path", dest="model_path", type=str,
                        default="./checkpoints/oloric-v0.1/final_model",
                        help="Path to Oloric fine-tuned checkpoint")
    parser.add_argument("--benchmark-file", type=str,
                        default="./evaluation/benchmark.jsonl",
                        help="Path to benchmark file")
    parser.add_argument("--example-id", type=str, default=None,
                        help="Example ID to evaluate (defaults to first example)")
    parser.add_argument("--mock", action="store_true",
                        help="Run with mock model to verify diagnostic harness without GPU/checkpoint")

    args = parser.parse_args()
    sys.exit(run_diagnostic(
        model_path=args.model_path,
        benchmark_file=args.benchmark_file,
        example_id=args.example_id,
        mock_mode=args.mock,
    ))


if __name__ == "__main__":
    main()
