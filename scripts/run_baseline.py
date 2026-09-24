#!/usr/bin/env python3
"""
Baseline inference script for OLORIC.
Runs the base model (without fine-tuning) to establish a baseline.
"""
import os
import sys
import json
import argparse
from typing import List, Dict, Any

# Ensure src directory is on sys.path
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_dir = os.path.join(_repo_root, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from oloric.inference import OloricInference
from oloric.schemas.model_input import OloricModelInput

def create_sample_input() -> OloricModelInput:
    """Create a sample input for testing."""
    from oloric.schemas.model_input import DocumentContext, LearnerState, ConversationTurn
    
    document_context = DocumentContext(
        document_id="econ101_chapter3",
        title="Principles of Economics",
        page=37,
        section="Multiplier",
        selected_text="The marginal propensity to consume (MPC)...",
        surrounding_context="In Keynesian economics, the multiplier effect demonstrates how an initial change in spending leads to a larger change in overall economic output.",
        retrieved_evidence=[
            "MPC = ΔC/ΔY where C is consumption and Y is income",
            "The multiplier is 1/(1-MPC)"
        ]
    )
    
    learner_state = LearnerState(
        level="beginner",
        concept="MPC",
        mastery=0.42,
        known_prerequisites=["basic algebra"],
        weak_prerequisites=["understanding of slopes"],
        known_misconceptions=[]
    )
    
    conversation_context = [
        ConversationTurn(
            role="student",
            content="I don't understand MPC."
        )
    ]
    
    current_goal = "Understand MPC well enough to apply it in economic calculations"
    
    return OloricModelInput(
        task="resolve_confusion",
        document_context=document_context,
        learner_state=learner_state,
        conversation_context=conversation_context,
        current_goal=current_goal
    )

def load_input_from_file(file_path: str) -> OloricModelInput:
    """Load model input from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return OloricModelInput(**data)

def save_output_to_file(output, file_path: str) -> None:
    """Save model output to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(output.dict(), f, indent=2)

def main():
    """Main baseline execution function."""
    parser = argparse.ArgumentParser(description="Run OLORIC baseline inference")
    parser.add_argument("--input", type=str, help="Path to input JSON file")
    parser.add_argument("--output", type=str, help="Path to output JSON file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Base model to use")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of sample inputs to process")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("OLORIC Baseline Inference")
    print("=" * 60)
    
    # Initialize inference engine with base model
    print(f"Loading base model: {args.model}")
    inference_engine = OloricInference()  # Uses base model by default
    
    # Prepare inputs
    inputs = []
    if args.input:
        # Load from file
        print(f"Loading input from: {args.input}")
        inputs.append(load_input_from_file(args.input))
    else:
        # Generate sample inputs
        print(f"Generating {args.num_samples} sample input(s)")
        for i in range(args.num_samples):
            inputs.append(create_sample_input())
    
    # Process inputs
    print("\nProcessing inputs...")
    outputs = []
    for i, model_input in enumerate(inputs):
        print(f"  Processing input {i+1}/{len(inputs)}")
        try:
            output = inference_engine.generate_response(model_input)
            outputs.append(output)
            
            # Print brief result
            print(f"    Action: {output.action}")
            print(f"    Strategy: {output.strategy}")
            print(f"    Response length: {len(output.response)} characters")
        except Exception as e:
            print(f"    Error processing input {i+1}: {e}")
    
    # Save outputs
    if args.output:
        if len(outputs) == 1:
            save_output_to_file(outputs[0], args.output)
            print(f"\nOutput saved to: {args.output}")
        else:
            # Save multiple outputs
            base, ext = os.path.splitext(args.output)
            for i, output in enumerate(outputs):
                output_file = f"{base}_{i+1}{ext}"
                save_output_to_file(output, output_file)
            print(f"\nOutputs saved to: {base}_*{ext}")
    else:
        # Print outputs to console
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        for i, output in enumerate(outputs):
            print(f"\nOutput {i+1}:")
            print(f"Action: {output.action}")
            print(f"Strategy: {output.strategy}")
            print(f"Difficulty: {output.difficulty}")
            print(f"Response:\n{output.response}")
            if output.understanding_check.required:
                print(f"Understanding Check: {output.understanding_check.question}")
            print(f"Diagnosis: {output.diagnosis.confusion_type}")
            if output.memory.candidate:
                print(f"Memory Candidate: {output.memory.title}")
    
    print("\n" + "=" * 60)
    print("Baseline inference complete!")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
