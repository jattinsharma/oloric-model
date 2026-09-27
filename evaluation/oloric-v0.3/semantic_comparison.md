# OLORIC Semantic Comparison: Baseline → v0.1 → v0.2 → v0.3

**Purpose**: Dimension-by-dimension cross-version comparison using the same 0–4 rubric.  
**Benchmark**: `evaluation/benchmark.jsonl` (20 items, frozen)  
**Rubric**: `evaluation/baseline/semantic_rubric.md`

---

## Aggregate Dimension Table

| Dimension | Baseline (general) | v0.1 | v0.2 | v0.3 |
|---|---|---|---|---|
| Factual Correctness | 2.75 | 0.00 | 2.30 | **3.65** |
| Simplicity | 2.85 | 1.00 | 2.70 | **3.05** |
| Teaching Strategy Selection | 2.10 | 1.21 | 2.05 | **3.70** |
| Strategy Switching | 2.00 | 1.33 | 2.67 | **3.00** |
| Misconception Detection | 1.50 | 1.20 | 1.40 | **4.00** |
| Prerequisite Detection | 2.00 | 1.00 | 1.33 | **2.50** |
| Context Retention | 2.10 | 1.26 | 2.05 | **2.40** |
| Diagnostic Question Quality | 2.00 | 1.00 | 2.00 | **3.25** |
| Understanding Check Quality | 2.20 | 1.00 | 2.63 | **2.81** |
| Memory Candidate Quality | 2.40 | 1.00 | 2.00 | **3.50** |
| Document Grounding | 1.10 | 0.00 | 0.95 | **1.15** |
| **Mean (scored dims)** | **2.09** | **0.92** | **2.10** | **2.96** |

> Baseline uses general (non-structured) mode. v0.1/v0.2/v0.3 use structured mode.
> Means exclude N/A dimensions per record.

---

## Trajectory Narrative

### v0.1 (Structured Mode Regression)
Switching to structured output mode initially collapsed all performance. Every content field was a bracket placeholder (`[detailed explanation]`, `[concrete example]`). Factual correctness: 0.00. The model learned the JSON schema shape but not the task-conditional content. Mean: 0.92.

### v0.2 (Content Recovery, Task-Type Collapse)
Pipeline improvements produced real content (factual=2.30) and eliminated placeholder text. However, 16/20 benchmark tasks selected `concrete_example/explain` regardless of the requested task type — a systematic task-following collapse. Cross-domain caloric bleed contaminated 7/20 responses with wrong-domain formulas. Misconception correction body was non-sequitur (identified misconception correctly but argued with unrelated content). Mean: 2.10.

### v0.3 (Targeted Recovery of v0.2 Failure Modes)
The v0.3 pipeline fix (injecting `task` + `instruction` into ChatML, preserving them through `evaluate_example()`) combined with task-specific training data resolves the four root causes:

1. **Task-type following**: 18/20 correctly matched (vs 4/20 in v0.2). Teaching Strategy Selection jumps from 2.05 → 3.70 (+1.65).
2. **Cross-domain caloric bleed**: Eliminated across all 7 affected items (FC-01 resolved).
3. **Misconception correction**: Both tasks achieve 4/4 with logically valid counterarguments (FC-04 resolved).
4. **Prerequisite detection**: Improved from 1.33 → 2.50; correct prerequisite identified.

FC-03 (prior-error propagation in hint_based_teaching) persists. Document grounding remains weak (1.15/4). Mean: 2.96.

---

## Dimension Deep Dives

### Teaching Strategy Selection (1.21 → 2.05 → 3.70)
The single largest improvement across all dimensions. In v0.1, strategy selection averaged 1.21 because every response used the same definition template regardless of task. In v0.2, the model produced real content but still defaulted to `concrete_example` in 13/20 cases. In v0.3, the pipeline fix and task-type conditioning allow correct selection of `diagnose`, `follow_up_questions`, `practice`, `prerequisite`, `hint`, `simple_explanation`, `misconception_correction` — matching the requested task in 18/20 cases.

### Misconception Detection (1.20 → 1.40 → 4.00)
The most dramatic improvement: +2.60 from v0.2 to v0.3. In v0.1, misconception_addressed fields were fabricated (e.g., "Organic food is always more nutritious" injected into non-misconception contexts). In v0.2, the correct misconception was identified but the correction was a non-sequitur (Linear Algebra as proof that not all continuous functions are differentiable; spaced repetition as proof of expert bias). In v0.3, both misconception tasks achieve 4/4 with valid, domain-appropriate counterarguments.

### Factual Correctness (0.00 → 2.30 → 3.65)
v0.1: zero content. v0.2: real content but cross-domain bleed (7 items with factually wrong examples attributed to wrong concepts). v0.3: concept-specific content with accurate numerical examples — catalase kinetics, mitotic doubling, monetary policy interest rate mechanics, food preservation canning temperatures.

### Diagnostic Question Quality (1.00 → 2.00 → 3.25)
v0.1: zero diagnostic questions produced. v0.2: some questions produced but generic (4 generic orientation questions for every task). v0.3: concept-targeted questions — `economics_diagnostic_questions_013` achieves 4/4 with a numerically operationalized MPC probe; the other diagnostic items score 3 (Q1-Q2 concept-targeted, Q3-Q4 generic).

### Memory Candidate Quality (1.00 → 2.00 → 3.50)
v0.1: placeholder memories ("Key example: [specific evidence]"). v0.2: real content but anchor-to-content mismatches (caloric formulas under Enzyme Activity anchor; Monetary Policy anchor with MPC formula). v0.3: accurate, well-anchored memories — Bias Identification memory contains a real editorial bias example; misconception memories contain the valid counterarguments.

### Document Grounding (0.00 → 0.95 → 1.15)
Consistent weakness across all versions. The retrieved evidence in the benchmark is mostly present in the ChatML prompt but almost never explicitly cited. Best grounding in v0.3: `economics_diagnostic_questions_013` operationalizes MPC=ΔC/ΔY formula in Q1; `accountancy_prerequisite_detection_014` grounds in Assets=Liabilities+Equity. All others score 1 (evidence present but unused).

### Understanding Check Quality (1.00 → 2.63 → 2.81)
Modest improvement. v0.2 had already improved dramatically from v0.1 by producing real comprehension questions. v0.3 further improves for the misconception detection items (4/4) and misconception items anchor the expected_answer to specific counterexample content. The remaining weakness: `follow_up_questions` and `diagnostic_questions` tasks use a generic meta-question template ("What is a good question to test deep understanding") — scores 2 rather than 3.

---

## Task-Type Resolution Matrix

| Task Type | # Items | v0.2 Success | v0.3 Success |
|---|---|---|---|
| concrete_example | 1 | 1/1 | 1/1 |
| simple_explanation | 1 | 1/1 | 1/1 |
| follow_up_questions | 4 | 1/4 | 4/4 |
| diagnostic_questions | 4 | 0/4 | 4/4 |
| numerical_example | 3 | 0/3 | 3/3 |
| misconception_detection | 2 | 2/2 (shape), 0/2 (content) | 2/2 |
| practice_questions | 1 | 0/1 (schema invalid) | 1/1 |
| hint_based_teaching | 1 | 1/1 (format), 0/1 (FC-03) | 1/1 (partial) |
| prerequisite_detection | 1 | 1/1 (borderline wrong) | 1/1 (correct) |
| **Total** | **18** | **~6/18 fully correct** | **~18/18 format, ~17/18 content** |

---

## Unresolved Issues for v0.4

| Issue | Severity | Instances |
|---|---|---|
| FC-03: Prior-error propagation in hint-based teaching | High | 1/20 |
| Understanding check meta-template (generic) | Medium | 6/20 |
| Document grounding — evidence not explicitly cited | Medium | 18/20 |
| Prerequisite scaffolding gap (identify but not bridge) | Low | 3/20 |
