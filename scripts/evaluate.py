#!/usr/bin/env python3
"""
Evaluation script for OLORIC.
Evaluates the trained model on benchmark datasets.
"""
import os
import sys
import json
import time
import argparse
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

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate OLORIC model")
    parser.add_argument("--model", "--model-path", dest="model_path", type=str, default=None,
                        help="Path to fine-tuned model (uses base model if not specified)")
    parser.add_argument("--benchmark-file", type=str, default="./evaluation/benchmark.jsonl",
                        help="Path to benchmark dataset")
    parser.add_argument("--output-dir", type=str, default="./evaluation/reports",
                        help="Output directory for evaluation reports")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples to evaluate")
    parser.add_argument("--compare-baseline", action="store_true",
                        help="Compare with base model")
    parser.add_argument("--baseline-model-path", type=str, default=None,
                        help="Path to baseline model for comparison")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("OLORIC Model Evaluation")
    print("=" * 60)
    
    # Initialize inference engine
    print("Loading model...")
    try:
        if args.model_path:
            print(f"Loading fine-tuned model from: {args.model_path}")
            inference_engine = OloricInference(model_path=args.model_path)
        else:
            print("Using base model for evaluation")
            inference_engine = OloricInference()  # Base model
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Initialize evaluator
    print("Initializing evaluator...")
    try:
        evaluator = OloricEvaluator(inference_engine=inference_engine)
        # Override benchmark path if specified
        if args.benchmark_file:
            evaluator.benchmark_path = args.benchmark_file
    except Exception as e:
        print(f"Error initializing evaluator: {e}")
        return 1
    
    # Run evaluation
    print("Running evaluation...")
    try:
        results = evaluator.evaluate_benchmark(limit=args.limit)
        print(f"Evaluated {len(results)} examples")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1
    
    # Generate report
    print("Generating report...")
    try:
        report = evaluator.generate_report(results)
        
        # Save report
        os.makedirs(args.output_dir, exist_ok=True)
        report_file = os.path.join(args.output_dir, f"evaluation_report_{int(time.time())}.json")
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Evaluation report saved to: {report_file}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Number of examples: {report['num_examples']}")
        print(f"Overall score: {report['overall_score']['mean']:.2f} "
              f"(range: {report['overall_score']['min']:.2f} - {report['overall_score']['max']:.2f})")
        
        print("\nDimension Scores:")
        for dim, scores in report['dimension_scores'].items():
            print(f"  {dim}: {scores['mean']:.2f} "
                  f"(range: {scores['min']:.2f} - {scores['max']:.2f})")
        
    except Exception as e:
        print(f"Error generating report: {e}")
        return 1
    
    # Baseline comparison if requested
    if args.compare_baseline:
        print("\nRunning baseline comparison...")
        try:
            # This would involve loading a baseline model and running evaluation
            # For now, we'll just note that this feature is planned
            print("Baseline comparison feature is planned for future implementation")
        except Exception as e:
            print(f"Error during baseline comparison: {e}")
    
    print("=" * 60)
    print("Evaluation completed!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    import time  # Import here to avoid issues if not used
    sys.exit(main())
