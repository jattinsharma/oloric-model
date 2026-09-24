import json

# Load benchmark file
with open('evaluation/benchmark.jsonl', 'r') as f:
    lines = f.readlines()

# Process each line
fixed_lines = []
for line_num, line in enumerate(lines, 1):
    line = line.strip()
    if not line:
        continue

    try:
        example = json.loads(line)

        # Add task field to target if missing
        if 'target' in example and 'task' not in example['target']:
            # Get task from context if available, otherwise default to resolve_confusion
            task = example.get('context', {}).get('task', 'resolve_confusion')
            example['target']['task'] = task

        fixed_lines.append(json.dumps(example))
    except json.JSONDecodeError as e:
        print(f"Error parsing line {line_num}: {e}")
        fixed_lines.append(line)  # Keep original line if error

# Write fixed benchmark file
with open('evaluation/benchmark.jsonl', 'w') as f:
    for line in fixed_lines:
        f.write(line + '\n')

print(f"Fixed {len(fixed_lines)} lines in benchmark.jsonl")