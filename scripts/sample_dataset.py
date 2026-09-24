#!/usr/bin/env python3
"""
Sample a subset of examples from a JSONL dataset while maintaining
distribution across domains and categories.
"""
import json
import os
import random
from collections import defaultdict

def load_dataset(file_path):
    """Load JSONL dataset."""
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples

def save_dataset(examples, file_path):
    """Save examples to JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

def main():
    input_file = "./data/generated_fixed/improved_seed_data.jsonl"
    output_file = "./data/generated_sampled/improved_seed_data_sampled.jsonl"

    print(f"Loading dataset from {input_file}...")
    examples = load_dataset(input_file)
    total_available = len(examples)
    print(f"Total examples available: {total_available}")

    # Group by (domain, category)
    groups = defaultdict(list)
    for example in examples:
        domain = example.get('domain')
        category = example.get('category')
        key = (domain, category)
        groups[key].append(example)

    print(f"Number of groups: {len(groups)}")

    # We want 480 examples total
    total_desired = 480
    # Calculate ideal count per group (proportional)
    ideal_counts = {}
    for key, group in groups.items():
        ideal_counts[key] = len(group) * total_desired / total_available

    # Assign integer counts using largest remainder method
    assigned_counts = {}
    remaining = total_desired
    # First assign the floor
    for key in groups:
        assigned_counts[key] = int(ideal_counts[key])
        remaining -= assigned_counts[key]

    # Distribute remaining to groups with largest fractional part
    fractional_parts = [(key, ideal_counts[key] - assigned_counts[key]) for key in groups]
    fractional_parts.sort(key=lambda x: x[1], reverse=True)

    for i in range(remaining):
        key = fractional_parts[i][0]
        assigned_counts[key] += 1

    # Verify total
    total_assigned = sum(assigned_counts.values())
    print(f"Total assigned examples: {total_assigned}")

    # Sample from each group
    random.seed(42)  # for reproducibility
    sampled_examples = []
    for key, group in groups.items():
        n = assigned_counts[key]
        if n > len(group):
            print(f"Warning: group {key} requested {n} but only {len(group)} available. Taking all.")
            n = len(group)
        sampled = random.sample(group, n)
        sampled_examples.extend(sampled)
        print(f"Group {key}: took {n} out of {len(group)}")

    # Shuffle the final dataset
    random.shuffle(sampled_examples)

    print(f"Saving {len(sampled_examples)} examples to {output_file}...")
    save_dataset(sampled_examples, output_file)

    # Quick stats
    domains_count = defaultdict(int)
    categories_count = defaultdict(int)
    for example in sampled_examples:
        domains_count[example.get('domain')] += 1
        categories_count[example.get('category')] += 1

    print("\nSampled dataset statistics:")
    print(f"Total examples: {len(sampled_examples)}")
    print("Examples per domain:")
    for domain in sorted(domains_count):
        print(f"  {domain}: {domains_count[domain]}")
    print("Examples per category:")
    for category in sorted(categories_count):
        print(f"  {category}: {categories_count[category]}")

    return 0

if __name__ == "__main__":
    exit(main())