#!/usr/bin/env python3
"""
Evaluation script for OLORIC.

Runs the trained model over the benchmark and persists every raw response
incrementally to <output-dir>/responses.jsonl — the input file for manual
semantic scoring. Generation settings are aligned 1:1 with the baseline
benchmark inference run (evaluation/baseline/config.json):
max_new_tokens=1024, temperature=0.3, top_p=0.9, do_sample=True, no seed.
"""
import os
import sys
import json
import time
import hashlib
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any

# Ensure src directory is on sys.path
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_dir = os.path.join(_repo_root, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from oloric.evaluation import OloricEvaluator
from oloric.inference import OloricInference


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _gpu_info() -> Dict[str, Any]:
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "name": props.name,
                "total_vram_gb": round(props.total_memory / (1024 ** 3), 2),
                "cuda_available": True,
            }
        return {"name": None, "total_vram_gb": None, "cuda_available": False}
    except Exception:
        return {"name": "unknown", "total_vram_gb": None, "cuda_available": False}


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate OLORIC model")
    parser.add_argument("--model", "--model-path", dest="model_path", type=str,
                        default="./checkpoints/oloric-v0.1/final_model",
                        help="Path to the trained Oloric adapter "
                             "(default: ./checkpoints/oloric-v0.1/final_model)")
    parser.add_argument("--benchmark-file", type=str, default="./evaluation/benchmark.jsonl",
                        help="Path to benchmark dataset")
    parser.add_argument("--output-dir", type=str, default="./evaluation/oloric-v0.1",
                        help="Output directory for evaluation artifacts")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples to evaluate")
    parser.add_argument("--compare-baseline", action="store_true",
                        help="Compare with base model")
    parser.add_argument("--baseline-model-path", type=str, default=None,
                        help="Path to baseline model for comparison")

    args = parser.parse_args()

    t_run_start = time.time()

    print("=" * 60)
    print("OLORIC Model Evaluation (Oloric v0.1)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Initialize inference engine (trained adapter)
    # ------------------------------------------------------------------
    print(f"Loading trained adapter from: {args.model_path}")
    t_model_start = time.time()
    try:
        inference_engine = OloricInference(model_path=args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    model_load_time = time.time() - t_model_start
    print(f"Model + adapter loaded in {model_load_time:.1f}s")

    # ------------------------------------------------------------------
    # Record the exact generation configuration for this run.
    # Mirrors the baseline benchmark run (evaluation/baseline/config.json):
    #   max_new_tokens=1024, temperature=0.3, top_p=0.9, do_sample=True.
    # The baseline set no random seed, so generation stays stochastic —
    # deterministic generation would NOT match the baseline configuration.
    # ------------------------------------------------------------------
    gen_cfg = inference_engine.generation_config
    model_id = (inference_engine.model_config.get("base_model", {}) or {}).get(
        "name", "Qwen/Qwen3-4B-Instruct-2507")
    run_config = {
        "model_id": model_id,
        "adapter_path": args.model_path,
        "adapter_present": bool(args.model_path and os.path.exists(args.model_path)),
        "generation_config": {
            "max_new_tokens": gen_cfg.max_new_tokens,
            "temperature": gen_cfg.temperature,
            "top_p": gen_cfg.top_p,
            "do_sample": bool(gen_cfg.do_sample),
            "seed": None,  # baseline set no seed; kept identical
        },
        "deterministic": False,  # do_sample=True and no seed -> stochastic (like baseline)
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "bfloat16",
        },
        "benchmark_file": args.benchmark_file,
        "benchmark_hash": _sha256_file(args.benchmark_file),
        "gpu": _gpu_info(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_load_time_seconds": round(model_load_time, 1),
        "baseline_reference": {
            "config_file": "evaluation/baseline/config.json",
            "max_new_tokens": 1024,
            "temperature": 0.3,
            "top_p": 0.9,
            "do_sample": True,
            "seed": None,
        },
    }

    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    print(f"Run config saved: {config_path}")

    # ------------------------------------------------------------------
    # Initialize evaluator
    # ------------------------------------------------------------------
    print("Initializing evaluator...")
    try:
        evaluator = OloricEvaluator(inference_engine=inference_engine)
        if args.benchmark_file:
            evaluator.benchmark_path = args.benchmark_file
    except Exception as e:
        print(f"Error initializing evaluator: {e}")
        return 1

    # ------------------------------------------------------------------
    # Run evaluation, writing each raw response to disk as soon as it is
    # produced so an interrupted run still preserves all completed
    # generations. No automated semantic scores are produced: overall_score
    # is always null and manual scoring happens from responses.jsonl.
    # ------------------------------------------------------------------
    print("Running evaluation...")
    responses_file = os.path.join(args.output_dir, "responses.jsonl")
    results = []
    last_ts = [None]

    try:
        with open(responses_file, "w", encoding="utf-8") as responses_fh:
            def _record_response(r):
                now = time.time()
                example_time = None
                if last_ts[0] is not None:
                    example_time = round(now - last_ts[0], 2)
                last_ts[0] = now
                entry = {
                    "id": r.example_id,
                    "mode": "structured",
                    "schema_valid": r.schema_valid,
                    "raw_response": r.raw_response,
                    "raw_generated_text": r.raw_generated_text,
                    "validation_error": r.validation_error,
                    "response_length": len(r.raw_response or ""),
                    "evaluation_time_seconds_approx": example_time,
                    "overall_score": r.overall_score,  # None: semantic scoring is manual
                }
                responses_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                responses_fh.flush()

            results = evaluator.evaluate_benchmark(limit=args.limit, on_result=_record_response)
        print(f"Evaluated {len(results)} examples")
        print(f"Raw responses saved (incremental): {responses_file}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1

    eval_time = time.time() - t_run_start - model_load_time
    total_time = time.time() - t_run_start

    # ------------------------------------------------------------------
    # Generate report (structural only; semantic scores are manual)
    # ------------------------------------------------------------------
    print("Generating report...")
    try:
        report = evaluator.generate_report(results)

        report_file = os.path.join(args.output_dir, f"evaluation_report_{int(time.time())}.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Evaluation report saved to: {report_file}")

        # ---- Final verification summary ----
        num_valid = sum(1 for r in results if r.schema_valid)
        num_invalid = sum(1 for r in results if not r.schema_valid)
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Responses written:      {len(results)}")
        print(f"  schema-valid:         {num_valid}")
        print(f"  schema-invalid:       {num_invalid}")
        print(f"Raw responses file:     {responses_file}")
        print(f"Evaluation time:        {eval_time:.1f}s "
              f"(total incl. model load: {total_time:.1f}s)")
        print(f"Generation settings:    max_new_tokens={gen_cfg.max_new_tokens} "
              f"temperature={gen_cfg.temperature} top_p={gen_cfg.top_p} "
              f"do_sample={gen_cfg.do_sample} seed=None")
        print("Deterministic:          NO (baseline also sampled unseeded at temp 0.3)")
        print("Automated semantic scores: NONE (manual scoring required; see "
              "evaluation/baseline/semantic_rubric.md)")
    except Exception as e:
        print(f"Error generating report: {e}")
        return 1

    # Baseline comparison if requested
    if args.compare_baseline:
        print("\nRunning baseline comparison...")
        try:
            print("Baseline comparison feature is planned for future implementation")
        except Exception as e:
            print(f"Error during baseline comparison: {e}")

    print("=" * 60)
    print("Evaluation completed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
