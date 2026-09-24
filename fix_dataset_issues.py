import json
import re

def fix_science_response(response, concept):
    """Fix science responses that incorrectly use F = ma"""
    # Replace incorrect F = ma usage with appropriate explanations
    if 'it is defined by F = ma' in response:
        # Provide concept-appropriate explanations
        concept_lower = concept.lower()
        if 'dna' in concept_lower or 'replication' in concept_lower:
            return response.replace('it is defined by F = ma', 'it involves the copying of DNA molecules to produce two identical replicas')
        elif 'mitosis' in concept_lower or 'cell' in concept_lower:
            return response.replace('it is defined by F = ma', 'it is the process of cell division that results in two genetically identical daughter cells')
        elif 'photosynthesis' in concept_lower:
            return response.replace('it is defined by F = ma', 'it is the process by which plants convert light energy into chemical energy')
        elif 'natural selection' in concept_lower:
            return response.replace('it is defined by F = ma', 'it is the process where organisms better adapted to their environment tend to survive and produce more offspring')
        else:
            # Generic science fix
            return response.replace('it is defined by F = ma', f'it is a fundamental concept in science related to {concept}')
    return response

def fix_economics_response(response, concept):
    """Fix economics responses that incorrectly use MPC = ΔC/ΔY"""
    if 'it is defined by MPC = ΔC/ΔY' in response:
        concept_lower = concept.lower()
        if 'gdp' in concept_lower:
            return response.replace('it is defined by MPC = ΔC/ΔY', 'it measures the total value of goods and services produced')
        elif 'inflation' in concept_lower:
            return response.replace('it is defined by MPC = ΔC/ΔY', 'it represents the rate at which the general level of prices for goods and services is rising')
        elif 'externalities' in concept_lower:
            return response.replace('it is defined by MPC = ΔC/ΔY', 'it refers to the side effects or consequences of industrial or commercial activities')
        elif 'supply and demand' in concept_lower:
            return response.replace('it is defined by MPC = ΔC/ΔY', 'it describes how prices vary based on the balance between product availability and demand')
        elif 'monetary policy' in concept_lower:
            return response.replace('it is defined by MPC = ΔC/ΔY', 'it involves the management of money supply and interest rates by central banks')
        elif 'fiscal policy' in concept_lower:
            return response.replace('it is defined by MPC = ΔC/ΔY', 'it involves government spending and tax policies to influence economic conditions')
        elif 'elasticity' in concept_lower:
            return response.replace('it is defined by MPC = ΔC/ΔY', 'it measures how much the quantity demanded or supplied responds to changes in price')
        elif 'comparative advantage' in concept_lower:
            return response.replace('it is defined by MPC = ΔC/ΔY', 'it describes the ability to produce goods or services at a lower opportunity cost than others')
        else:
            # Generic economics fix
            return response.replace('it is defined by MPC = ΔC/ΔY', f'it is an important economic concept related to {concept}')
    return response

def fix_mathematics_response(response, concept):
    """Fix mathematics responses that incorrectly use quadratic formula"""
    if 'quadratic formula: x = (-b ± √(b²-4ac))/2a' in response:
        concept_lower = concept.lower()
        if 'integral' in concept_lower or 'integrals' in concept_lower:
            return response.replace('quadratic formula: x = (-b ± √(b²-4ac))/2a', 'it represents the area under a curve or the accumulation of quantities')
        elif 'exponential' in concept_lower:
            return response.replace('quadratic formula: x = (-b ± √(b²-4ac))/2a', 'it describes growth or decay at a constant percentage rate')
        elif 'logarithm' in concept_lower or 'logarithms' in concept_lower:
            return response.replace('quadratic formula: x = (-b ± √(b²-4ac))/2a', 'it is the inverse operation to exponentiation')
        elif 'quadratic' in concept_lower:
            # Keep quadratic formula for quadratic equations
            return response
        else:
            # Generic mathematics fix
            return response.replace('quadratic formula: x = (-b ± √(b²-4ac))/2a', f'it is a fundamental mathematical concept related to {concept}')
    return response

def fix_accountancy_response(response, concept):
    """Fix accountancy responses that incorrectly use Assets = Liabilities + Equity"""
    if 'it is defined by Assets = Liabilities + Equity' in response:
        concept_lower = concept.lower()
        if 'audit' in concept_lower:
            return response.replace('it is defined by Assets = Liabilities + Equity', 'it involves the examination and verification of financial records')
        elif 'depreciation' in concept_lower:
            return response.replace('it is defined by Assets = Liabilities + Equity', 'it is the allocation of the cost of a tangible asset over its useful life')
        elif 'tax' in concept_lower:
            return response.replace('it is defined by Assets = Liabilities + Equity', 'it involves the processes and regulations related to government taxation')
        elif 'cost-volume-profit' in concept_lower:
            return response.replace('it is defined by Assets = Liabilities + Equity', 'it examines how changes in costs and volume affect operating profit')
        else:
            # Generic accountancy fix
            return response.replace('it is defined by Assets = Liabilities + Equity', f'it is an important accounting concept related to {concept}')
    return response

def process_example(example):
    """Process a single example to fix issues"""
    target = example.get('target', {})
    if not target:
        return example

    response = target.get('response', '')
    concept = example.get('context', {}).get('learner_state', {}).get('concept', '')
    domain = example.get('domain', '')

    # Fix based on domain
    if domain == 'science' or domain == 'nutrition_food_science':
        new_response = fix_science_response(response, concept)
    elif domain == 'economics':
        new_response = fix_economics_response(response, concept)
    elif domain == 'mathematics':
        new_response = fix_mathematics_response(response, concept)
    elif domain == 'accountancy':
        new_response = fix_accountancy_response(response, concept)
    else:
        new_response = response

    # Update the response if it changed
    if new_response != response:
        target['response'] = new_response
        example['target'] = target

    return example

def main():
    input_path = 'data/generated_sampled/improved_seed_data_sampled.jsonl'
    output_path = 'data/generated_sampled/improved_seed_data_fixed.jsonl'

    fixed_examples = []

    with open(input_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
                fixed_example = process_example(example)
                fixed_examples.append(fixed_example)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue

    # Write fixed dataset
    with open(output_path, 'w') as f:
        for example in fixed_examples:
            f.write(json.dumps(example) + '\n')

    print(f'Fixed {len(fixed_examples)} examples')
    print(f'Original: {input_path}')
    print(f'Fixed: {output_path}')

    # Run validation on fixed dataset
    import subprocess
    result = subprocess.run([
        'python', 'scripts/validate_seed_data.py',
        '--data-path', output_path
    ], capture_output=True, text=True, cwd='C:\\oloric v01\\oloric-model')

    print('\\nValidation results:')
    print(result.stdout)
    if result.stderr:
        print('Validation errors:')
        print(result.stderr)

if __name__ == '__main__':
    main()