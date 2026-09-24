# OLORIC Evaluation Framework

This document describes the evaluation framework for the OLORIC adaptive tutoring model, including benchmark structure, deterministic automated scoring, and the 11-dimension semantic rubric.

---

## 1. Overview

Evaluation is structured into two complementary layers:

1. **Automated Structural Evaluation**: Deterministic checks for output schema conformance, required keys, JSON parsability, enum validity, and action consistency.
2. **Semantic Rubric Evaluation**: Qualitative pedagogical review across 11 key dimensions comparing tutor decisions against ground truth documents and student states.

---

## 2. Benchmark Dataset

- **Path**: `evaluation/benchmark.jsonl`
- **Total Scenarios**: 20 carefully curated scenarios
- **Coverage**:
  - Simple explanation & simplification
  - Analogy & concrete examples
  - Prerequisite & misconception detection
  - Multi-turn tutoring & strategy switching
  - Document grounding & context retention
  - Oloric Memory candidate generation

---

## 3. Evaluation Scripts & Usage

### 3.1 Baseline Inference (Gate 1)
```bash
python scripts/run_baseline_inference.py --mode both --benchmark evaluation/benchmark.jsonl --output-dir evaluation/baseline
```
- **Mode A (General)**: standard tutor prompt
- **Mode B (Structured)**: structured JSON schema prompt

### 3.2 Automated Structural Scoring
```bash
python scripts/score_baseline_automated.py \
    --responses evaluation/baseline/responses.jsonl \
    --output-dir evaluation/baseline
```
Metrics evaluated:
- JSON parse rate (%)
- Top-level schema compliance (%)
- Action field validity (%)
- Difficulty enum validity (%)
- Confusion type validity (%)
- Understanding check field completeness (%)
- Memory candidate structure validity (%)
- Mean response length (characters and tokens)

### 3.3 Semantic Rubric
The qualitative scoring rubric is detailed in [evaluation/baseline/semantic_rubric.md](file:///c:/oloric%20v01/oloric-model/evaluation/baseline/semantic_rubric.md).

The 11 evaluated dimensions:
1. `instruction_following`
2. `pedagogical_appropriateness`
3. `concept_clarity`
4. `document_fidelity`
5. `grounding_accuracy`
6. `strategy_adaptation`
7. `prerequisite_awareness`
8. `misconception_handling`
9. `conciseness`
10. `tone_and_engagement`
11. `memory_utility`

### 3.4 Fine-Tuned Model Evaluation
```bash
python scripts/evaluate.py \
    --model-path ./checkpoints/final_model \
    --benchmark-file ./evaluation/benchmark.jsonl \
    --output-dir ./evaluation/reports
```
