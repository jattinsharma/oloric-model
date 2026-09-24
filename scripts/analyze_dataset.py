#!/usr/bin/env python3
"""
Analysis script for OLORIC seed dataset.
Computes statistics about the generated dataset.
"""
import json
import os
from collections import Counter, defaultdict

def load_dataset(file_path):
    """Load JSONL dataset."""
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples

def analyze_dataset(examples):
    """Analyze dataset and compute statistics."""
    stats = {
        'total_examples': len(examples),
        'categories': {},
        'domains': {},
        'conversation_turns': [],
        'multi_turn_count': 0,
        'strategy_switch_count': 0,
        'misconception_count': 0,
        'prerequisite_count': 0,
        'understanding_check_count': 0,
        'memory_candidate_count': 0
    }

    # Count by category and domain
    for example in examples:
        category = example['category']
        domain = example['domain']

        stats['categories'][category] = stats['categories'].get(category, 0) + 1
        stats['domains'][domain] = stats['domains'].get(domain, 0) + 1

        # Analyze conversation turns
        conversation = example.get('context', {}).get('conversation_context', [])
        turn_count = len(conversation)
        stats['conversation_turns'].append(turn_count)
        if turn_count > 1:
            stats['multi_turn_count'] += 1

        # Check for strategy switching
        if example.get('target', {}).get('strategy') == 'analogy':
            # This is a simplification - in reality we'd need to check if strategy changed
            # For now, we'll count multi_turn_tutoring and strategy_switching categories
            pass

        # Count by category for specific types
        if category == 'strategy_switching':
            stats['strategy_switch_count'] += 1
        elif category == 'misconception_detection':
            stats['misconception_count'] += 1
        elif category == 'prerequisite_detection':
            stats['prerequisite_count'] += 1
        elif category == 'understanding_confirmation':
            stats['understanding_check_count'] += 1

        # Check for memory candidate
        if example.get('target', {}).get('memory', {}).get('candidate', False):
            stats['memory_candidate_count'] += 1

    # Compute averages and percentages
    if stats['conversation_turns']:
        stats['avg_conversation_turns'] = sum(stats['conversation_turns']) / len(stats['conversation_turns'])
    else:
        stats['avg_conversation_turns'] = 0

    stats['multi_turn_percentage'] = (stats['multi_turn_count'] / stats['total_examples']) * 100 if stats['total_examples'] > 0 else 0
    stats['strategy_switch_percentage'] = (stats['strategy_switch_count'] / stats['total_examples']) * 100 if stats['total_examples'] > 0 else 0
    stats['misconception_percentage'] = (stats['misconception_count'] / stats['total_examples']) * 100 if stats['total_examples'] > 0 else 0
    stats['prerequisite_percentage'] = (stats['prerequisite_count'] / stats['total_examples']) * 100 if stats['total_examples'] > 0 else 0
    stats['understanding_check_percentage'] = (stats['understanding_check_count'] / stats['total_examples']) * 100 if stats['total_examples'] > 0 else 0
    stats['memory_candidate_percentage'] = (stats['memory_candidate_count'] / stats['total_examples']) * 100 if stats['total_examples'] > 0 else 0

    return stats

def print_statistics(stats):
    """Print statistics in a readable format."""
    print("=" * 60)
    print("OLORIC SEED DATASET ANALYSIS")
    print("=" * 60)

    print(f"\nTotal Examples: {stats['total_examples']}")

    print(f"\nExamples per Category:")
    for category, count in sorted(stats['categories'].items()):
        print(f"  {category}: {count}")

    print(f"\nExamples per Domain:")
    for domain, count in sorted(stats['domains'].items()):
        print(f"  {domain}: {count}")

    print(f"\nConversation Statistics:")
    print(f"  Average turns per conversation: {stats['avg_conversation_turns']:.2f}")
    print(f"  Multi-turn conversations: {stats['multi_turn_count']} ({stats['multi_turn_percentage']:.1f}%)")

    print(f"\nSpecialized Example Types:")
    print(f"  Strategy switching: {stats['strategy_switch_count']} ({stats['strategy_switch_percentage']:.1f}%)")
    print(f"  Misconception detection: {stats['misconception_count']} ({stats['misconception_percentage']:.1f}%)")
    print(f"  Prerequisite detection: {stats['prerequisite_count']} ({stats['prerequisite_percentage']:.1f}%)")
    print(f"  Understanding confirmation: {stats['understanding_check_count']} ({stats['understanding_check_percentage']:.1f}%)")
    print(f"  Memory candidates: {stats['memory_candidate_count']} ({stats['memory_candidate_percentage']:.1f}%)")

def main():
    """Main analysis function."""
    dataset_path = "./data/generated/enhanced_seed_data.jsonl"

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return 1

    print(f"Loading dataset from {dataset_path}...")
    examples = load_dataset(dataset_path)
    print(f"Loaded {len(examples)} examples")

    stats = analyze_dataset(examples)
    print_statistics(stats)

    return 0

if __name__ == "__main__":
    exit(main())