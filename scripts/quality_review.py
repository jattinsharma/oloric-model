#!/usr/bin/env python3
"""
Quality review script for OLORIC seed dataset.
Outputs examples for manual quality review.
"""
import json
import os
import random
from typing import List, Dict, Any

def load_dataset(file_path):
    """Load JSONL dataset."""
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples

def select_random_sample(examples, sample_size=30):
    """Select a random sample of examples for review."""
    return random.sample(examples, min(sample_size, len(examples)))

def format_example_for_review(example: Dict[str, Any]) -> str:
    """Format an example for quality review."""
    output = []
    output.append("=" * 80)
    output.append(f"ID: {example.get('id', 'N/A')}")
    output.append(f"Category: {example.get('category', 'N/A')}")
    output.append(f"Domain: {example.get('domain', 'N/A')}")
    output.append(f"Instruction: {example.get('instruction', 'N/A')}")
    output.append("-" * 80)
    output.append("Context:")
    context = example.get('context', {})
    if context:
        output.append(f"  Task: {context.get('task', 'N/A')}")
        doc_context = context.get('document_context', {})
        if doc_context:
            output.append(f"  Document: {doc_context.get('title', 'N/A')} (p.{doc_context.get('page', 'N/A')})")
            output.append(f"  Section: {doc_context.get('section', 'N/A')}")
            output.append(f"  Selected text: {doc_context.get('selected_text', 'N/A')[:100]}...")
        learner_state = context.get('learner_state', {})
        if learner_state:
            output.append(f"  Learner level: {learner_state.get('level', 'N/A')}")
            output.append(f"  Concept: {learner_state.get('concept', 'N/A')}")
            output.append(f"  Mastery: {learner_state.get('mastery', 'N/A')}")
        conversation = context.get('conversation_context', [])
        if conversation:
            output.append(f"  Conversation turns: {len(conversation)}")
            for i, turn in enumerate(conversation[:3]):  # Show first 3 turns
                output.append(f"    {turn.get('role', 'N/A')}: {turn.get('content', 'N/A')[:100]}...")
            if len(conversation) > 3:
                output.append(f"    ... and {len(conversation) - 3} more turns")
        output.append(f"  Current goal: {context.get('current_goal', 'N/A')}")
    output.append("-" * 80)
    output.append("Target:")
    target = example.get('target', {})
    if target:
        output.append(f"  Action: {target.get('action', 'N/A')}")
        output.append(f"  Strategy: {target.get('strategy', 'N/A')}")
        output.append(f"  Difficulty: {target.get('difficulty', 'N/A')}")
        output.append(f"  Response: {target.get('response', 'N/A')[:200]}...")
        understanding_check = target.get('understanding_check', {})
        if understanding_check:
            output.append(f"  Understanding check:")
            output.append(f"    Required: {understanding_check.get('required', 'N/A')}")
            output.append(f"    Question: {understanding_check.get('question', 'N/A')}")
            output.append(f"    Expected answer: {understanding_check.get('expected_answer', 'N/A')}")
        diagnosis = target.get('diagnosis', {})
        if diagnosis:
            output.append(f"  Diagnosis:")
            output.append(f"    Confusion type: {diagnosis.get('confusion_type', 'N/A')}")
            output.append(f"    Severity: {diagnosis.get('severity', 'N/A')}")
            output.append(f"    Misconception addressed: {diagnosis.get('misconception_addressed', 'N/A')}")
        memory = target.get('memory', {})
        if memory:
            output.append(f"  Memory:")
            output.append(f"    Candidate: {memory.get('candidate', 'N/A')}")
            output.append(f"    Title: {memory.get('title', 'N/A')}")
            output.append(f"    Content: {memory.get('content', 'N/A')[:100] if memory.get('content') else 'N/A'}...")
            output.append(f"    Anchor concept: {memory.get('anchor_concept', 'N/A')}")
            output.append(f"    Memory type: {memory.get('memory_type', 'N/A')}")
            output.append(f"    Confidence: {memory.get('confidence', 'N/A')}")
    output.append("=" * 80)
    return "\n".join(output)

def main():
    """Main quality review function."""
    import argparse

    parser = argparse.ArgumentParser(description="Quality review for OLORIC seed dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/generated/enhanced_seed_data.jsonl",
        help="Path to dataset JSONL file"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30,
        help="Number of examples to review"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file for review (if not specified, prints to stdout)"
    )

    args = parser.parse_args()

    # Set random seed for reproducible sampling
    random.seed(args.seed)

    if not os.path.exists(args.dataset):
        print(f"Error: Dataset not found at {args.dataset}")
        return 1

    print(f"Loading dataset from {args.dataset}...")
    examples = load_dataset(args.dataset)
    print(f"Loaded {len(examples)} examples")

    print(f"Selecting random sample of {args.sample_size} examples...")
    sample = select_random_sample(examples, args.sample_size)

    # Prepare output
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("QUALITY REVIEW SAMPLE")
    output_lines.append("=" * 80)
    output_lines.append("For each example, classify as:")
    output_lines.append("  EXCELLENT: High quality, meets all requirements")
    output_lines.append("  ACCEPTABLE: Good quality, minor issues")
    output_lines.append("  WEAK: Poor quality, significant issues")
    output_lines.append("  INVALID: Does not meet basic requirements")
    output_lines.append("=" * 80)

    for i, example in enumerate(sample, 1):
        output_lines.append(f"\nExample {i}/{len(sample)}:")
        output_lines.append(format_example_for_review(example))

    output_lines.append("\n" + "=" * 80)
    output_lines.append("QUALITY REVIEW COMPLETE")
    output_lines.append("=" * 80)
    output_lines.append("Note: Please manually classify each example as E/A/W/I based on the criteria above.")
    output_lines.append("In a production system, these classifications would be saved for further analysis.")

    output_text = "\n".join(output_lines)

    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"Review sample saved to {args.output_file}")
    else:
        print(output_text)

    return 0

if __name__ == "__main__":
    exit(main())