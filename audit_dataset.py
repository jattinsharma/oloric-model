#!/usr/bin/env python3
"""
Audit the OLORIC seed dataset for depth and behavior.
"""
import json
import os
import random
from collections import Counter

def load_dataset(file_path):
    """Load JSONL dataset."""
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples

def classify_example(example):
    """Classify an example into categories for depth and behavior."""
    context = example.get('context', {})
    conversation = context.get('conversation_context', [])
    n = len(conversation)

    # Determine if single-turn or multi-turn based on conversation history length
    if n == 1:
        turn_type = 'single-turn'
    else:
        turn_type = 'multi-turn'

    # Further classify single-turn examples
    if turn_type == 'single-turn':
        # Check if the tutor response (in target) is a useful explanation
        target = example.get('target', {})
        response = target.get('response', '')
        # Heuristic: if the response contains placeholders like [explanation], it's shallow
        if '[explanation]' in response or '[reason]' in response or '[example]' in response:
            single_turn_subtype = 'shallow single-turn'
        else:
            single_turn_subtype = 'useful single-turn'
        return turn_type, single_turn_subtype
    else:
        # Multi-turn: check for strong Oloric trajectory
        # Look for signs of adaptive teaching in conversation and target
        strong_indicators = [
            "I still don't understand",
            "I still don't get it",
            "Let me try a different approach",
            "Think of it like",
            "Just as",
            "This helps us understand",
            "What prerequisite",
            "You need to know about",
            "A common misconception is",
            "This is incorrect because",
            "To check your understanding",
            "Consider these questions",
            "Great question!",
            "From our discussion:",
            "Remember:",
            "Key example:",
            "This illustrates",
            "Another example:",
            "Let's practice",
            "Try solving these",
            "I notice there's a misunderstanding",
            "Let me correct this",
            "You've got part of it right",
            "However, you're missing",
            "Excellent work!",
            "You've demonstrated solid understanding",
            "This shows you've grasped",
            "How would you explain",
            "I would explain it as",
            "I notice there's a misunderstanding in your work",
            "Let me correct this:",
            "Remember: [correct explanation], NOT [incorrect idea]",
            "You now understand",
            "well enough to"
        ]

        # Check conversation history for student expressing continued confusion
        student_turns = [turn for turn in conversation if turn.get('role') == 'student']
        continued_confusion = any(
            phrase in turn.get('content', '').lower()
            for turn in student_turns
            for phrase in ["still don't understand", "still don't get it", "i'm still confused"]
        )

        # Check target response for adaptive teaching indicators
        target = example.get('target', {})
        response = target.get('response', '')
        strategy = target.get('strategy', '')

        # Check if the strategy is different from what might have been used before?
        # We don't have the previous strategy, but we can look at the conversation history for tutor turns
        tutor_turns = [turn for turn in conversation if turn.get('role') == 'tutor']
        # If there are tutor turns, we cannot easily know the strategy from the content, so we skip.

        # Instead, we rely on the category and the presence of indicators
        strong_trajectory = continued_confusion or any(indicator in response for indicator in strong_indicators)

        # Also, if the example is in a category that implies strong trajectory, we can mark it as such
        category = example.get('category', '')
        strong_categories = {
            'strategy_switching', 'repeated_confusion', 'multi_turn_tutoring',
            'prerequisite_detection', 'misconception_detection',
            'understanding_confirmation', 'memory_generation'
        }
        if category in strong_categories:
            strong_trajectory = True

        if strong_trajectory:
            return turn_type, 'strong Oloric trajectory'
        else:
            return turn_type, 'weak multi-turn'

def audit_dataset(dataset_path):
    """Perform the audit on the dataset."""
    print(f"Loading dataset from {dataset_path}...")
    examples = load_dataset(dataset_path)
    total = len(examples)
    print(f"Total examples: {total}")

    # Overall classification
    turn_type_counts = Counter()
    subtype_counts = Counter()

    for example in examples:
        turn_type, subtype = classify_example(example)
        turn_type_counts[turn_type] += 1
        subtype_counts[subtype] += 1

    print("\n=== Turn Type Classification ===")
    for turn_type, count in turn_type_counts.items():
        print(f"{turn_type}: {count} ({count/total*100:.1f}%)")

    print("\n=== Detailed Subtype Classification ===")
    for subtype, count in sorted(subtype_counts.items()):
        print(f"{subtype}: {count} ({count/total*100:.1f}%)")

    # Now, inspect specific categories
    categories_to_inspect = [
        'strategy_switching', 'misconception_detection', 'prerequisite_detection',
        'understanding_confirmation', 'memory_generation'
    ]

    print("\n=== Category-Specific Inspection ===")
    for category in categories_to_inspect:
        cat_examples = [ex for ex in examples if ex.get('category') == category]
        print(f"\n{category} (total: {len(cat_examples)}):")
        weak_in_cat = []
        for i, example in enumerate(cat_examples[:10]):  # Inspect first 10 for brevity
            turn_type, subtype = classify_example(example)
            if 'weak' in subtype or 'shallow' in subtype:
                weak_in_cat.append((i, example.get('id'), subtype))
        print(f"  Inspected first 10: {len(weak_in_cat)} weak/shallow")
        if weak_in_cat:
            print("  Weak/shallow examples:")
            for idx, eid, sub in weak_in_cat[:5]:
                print(f"    {eid}: {sub}")

    # Count memory-candidate examples (not a category, but a flag in target)
    memory_candidate_count = sum(
        1 for ex in examples
        if ex.get('target', {}).get('memory', {}).get('candidate', False)
    )
    print(f"\nMemory-candidate examples (target.memory.candidate == True): {memory_candidate_count} ({memory_candidate_count/total*100:.1f}%)")

    # Calculate multi-turn percentage (based on our definition: n>=2)
    multi_turn_count = turn_type_counts.get('multi-turn', 0)
    print(f"\nMulti-turn examples (conversation_history length >= 2): {multi_turn_count} ({multi_turn_count/total*100:.1f}%)")

    # Average conversation turns
    total_turns = sum(len(ex.get('context', {}).get('conversation_context', [])) for ex in examples)
    avg_turns = total_turns / total if total > 0 else 0
    print(f"Average conversation turns in history: {avg_turns:.2f}")

    # Return stats for reporting
    stats = {
        'total_examples': total,
        'single_turn_count': turn_type_counts.get('single-turn', 0),
        'multi_turn_count': multi_turn_count,
        'avg_turns': avg_turns,
        'subtype_counts': dict(subtype_counts),
        'category_counts': {cat: len([ex for ex in examples if ex.get('category')==cat]) for cat in categories_to_inspect},
        'memory_candidate_count': memory_candidate_count
    }
    return stats, examples

def main():
    import sys
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = "./data/generated/enhanced_seed_data.jsonl"
    stats, examples = audit_dataset(dataset_path)

    # Save stats for later use
    import json
    with open("./data/generated/audit_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\nAudit stats saved to ./data/generated/audit_stats.json")

if __name__ == "__main__":
    main()