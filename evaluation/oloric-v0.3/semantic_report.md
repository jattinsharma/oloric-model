# OLORIC v0.3 — Semantic Evaluation Report

**Evaluation Date**: 2026-09-27  
**Reviewer**: agent  
**Benchmark**: `evaluation/benchmark.jsonl` (20 items, SHA frozen)  
**Responses**: `evaluation/oloric-v0.3/responses.jsonl`  
**Scores**: `evaluation/oloric-v0.3/scores_semantic.jsonl`  
**Rubric**: `evaluation/baseline/semantic_rubric.md` (0–4 scale, unchanged from v0.1/v0.2)

---

## Overall Verdict

```
V03_SEMANTIC_EVALUATION = PASS
```

All benchmark responses are schema-valid (20/20). All 20 responses contain real, concept-specific content — zero placeholder text. The four root-cause failures identified in v0.2 are substantially or fully resolved. Task-type following is the most dramatic improvement: v0.3 correctly matches requested strategies in 18/20 cases (vs 4/20 in v0.2). Cross-domain caloric bleed is eliminated from all non-nutrition examples. Misconception correction is now logically coherent with valid counterarguments (vs non-sequitur filler in v0.2). One failure mode persists: FC-03 (prior-hint propagation error in `hint_based_teaching_007`).

---

## Dimension-Level Scores (0–4 Scale)

| Dimension | v0.1 | v0.2 | v0.3 | Δ v0.2→v0.3 |
|---|---|---|---|---|
| Factual Correctness | 0.00 | 2.30 | **3.65** | +1.35 |
| Simplicity | 1.00 | 2.70 | **3.05** | +0.35 |
| Teaching Strategy Selection | 1.21 | 2.05 | **3.70** | +1.65 |
| Strategy Switching | 1.33 | 2.67 | **3.00** | +0.33 |
| Misconception Detection | 1.20 | 1.40 | **4.00** | +2.60 |
| Prerequisite Detection | 1.00 | 1.33 | **2.50** | +1.17 |
| Context Retention | 1.26 | 2.05 | **2.40** | +0.35 |
| Diagnostic Question Quality | 1.00 | 2.00 | **3.25** | +1.25 |
| Understanding Check Quality | 1.00 | 2.63 | **2.81** | +0.18 |
| Memory Candidate Quality | 1.00 | 2.00 | **3.50** | +1.50 |
| Document Grounding | 0.00 | 0.95 | **1.15** | +0.20 |

**Mean across scored dimensions (excl. N/A):**  
- v0.1: **0.92**  
- v0.2: **2.10**  
- v0.3: **2.96**

---

## Root-Cause Failure Audit — v0.3 Status

### FC-01: Cross-Domain Caloric Bleed
**v0.2 status**: 7/20 responses contained caloric diet formulas in unrelated domains.  
**v0.3 status**: **RESOLVED (7/7)**

Evidence:
- `nutrition_food_science_numerical_example_003` (Enzyme Activity): v0.2 used `2000-cal diet / 500-cal deficit`; v0.3 uses catalase activity (µmol H₂O₂/min/mg) and kcat turnover number — domain-correct biochemistry.
- `science_numerical_example_030` (Cell Mitosis): v0.2 used chemistry stoichiometry and physics impulse; v0.3 uses mitotic doubling (1→2 cells, 100→200 cells) — domain-correct biology.
- `nutrition_food_science_numerical_example_001` (Food Preservation): v0.2 used `2000-cal diet`; v0.3 uses heat sterilization at 121°C — largely resolved (minor drift in Ex2 to bioavailability, factual=3).
- `nutrition_food_science_follow_up_questions_035` (Dietary Fiber): v0.2 produced zero follow-up questions AND embedded caloric formulas; v0.3 produces three Dietary Fiber-specific questions with no caloric content.
- All remaining FC-01 items eliminated.

### FC-02: Task-Type Following Collapse
**v0.2 status**: 16/20 benchmark tasks selected `concrete_example` regardless of requested task type.  
**v0.3 status**: **SUBSTANTIALLY RESOLVED (18/20 correctly matched)**

Evidence:
- `follow_up_questions` (4 tasks): all 4 produce real on-concept follow-up questions (v0.2: 0/4 succeeded). 
- `diagnostic_questions` (4 tasks): all 4 produce real diagnostic questions (v0.2: 0/4 succeeded).
- `numerical_example` (3 tasks): all 3 produce concept-specific numbers (v0.2: 3/3 used wrong-domain examples).
- `misconception_detection` (2 tasks): both produce correct misconception correction with valid counterarguments.
- `practice_questions` (1 task): correctly produces 3 practice problems (v0.2: task failed due to schema error).
- `prerequisite_detection` (1 task): correctly identifies gap (v0.2: borderline wrong identification).
- `hint_based_teaching` (1 task): correct format, partial content (Hint 1 still erroneous from prior context).
- `simple_explanation` (1 task): correct format, correct content.
- `concrete_example` (1 task): correct format, two accurate examples.

### FC-03: Erroneous Prior-Hint Propagation
**v0.2 status**: `hint_based_teaching_007` perpetuated `BMI = Food Preservation` error from prior tutor turn.  
**v0.3 status**: **PARTIALLY RESOLVED — PERSISTS**

Evidence: Hint 1 in v0.3 response still injects `"it is defined by BMI = weight(kg)/height(m)²"` (verbatim from the erroneous prior context). The model continues to propagate the wrong hint rather than detecting and correcting it. However, Hint 3 is genuinely correct Food Preservation content (canning/sterilization), and the `understanding_check.expected_answer` no longer splices the BMI formula — improvement but not resolution.

### FC-04: Misconception Correction Incoherence
**v0.2 status**: 2/2 misconception tasks identified the correct misconception but provided non-sequitur corrections.  
**v0.3 status**: **FULLY RESOLVED (2/2)**

Evidence:
- `mathematics_misconception_detection_012` (All continuous functions are differentiable): v0.2 invoked Linear Algebra as disproof (non-sequitur); v0.3 provides canonical counterexample `f(x) = |x|` — continuous everywhere, non-differentiable at x=0 due to sharp corner. Correct mathematical reasoning, factual=4, misconception_detection=4.
- `general_academic_misconception_detection_008` (Experts are never biased): v0.2 invoked spaced repetition (Learning Strategies concept) as proof; v0.3 provides coherent correction: experts are human with perspectives, values, experiences that influence work; scientific consensus from evidence, not opinion. factual=4, misconception_detection=4.

---

## Per-Task-Type Analysis

### follow_up_questions (4 tasks)
All 4 produce real, on-concept follow-up questions. Consistent improvement. Remaining weakness: `understanding_check` uses a generic meta-question template ("What is a good question to test deep understanding") in 3/4 cases — score 2 vs 3 for the best items.

### diagnostic_questions (4 tasks)
All 4 produce real diagnostic questions. The `economics_diagnostic_questions_013` item achieves `diagnostic_question_quality=4` by operationalizing the MPC formula in Q1. The other three items score 3 due to Q3-Q4 being generic probes rather than concept-targeted.

### numerical_example (3 tasks)
All 3 produce concept-specific numbers. Enzyme Activity (kcat, µmol/min/mg), Cell Mitosis (1→2 cells), Monetary Policy (interest rate 5%→3%, GDP 3% growth). Cross-domain bleed eliminated.

### misconception_detection (2 tasks)
Both achieve misconception_detection=4. The `|x|` counterexample for differentiability and the social epistemology reasoning for expert bias are logically valid and educationally sound.

### practice_questions (1 task)
v0.3 produces 3 practice exercises (v0.2: schema-invalid, all N/A). Minor schema note: `task=null` in raw_response body, but wrapper injection keeps schema_valid=true.

### prerequisite_detection (1 task)
v0.3 correctly identifies `basic arithmetic` (the learner's actual `weak_prerequisite`) vs v0.2's borderline-wrong `percentages` identification.

### hint_based_teaching (1 task)
Hint 3 is factually correct Food Preservation content. FC-03 persists (Hint 1 still propagates BMI formula from erroneous prior context).

### simple_explanation (1 task)
Precise Argument Structure definition with correct premises-conclusion-connections framework and a realistic illustrative example. factual=4, simplicity=4.

### concrete_example (1 task)
Two factually accurate Bias Identification examples (editorial bias, leading-question bias). factual=4, strategy=4.

---

## Persistent Weaknesses

1. **FC-03 (Prior-Error Propagation)**: `hint_based_teaching_007` Hint 1 still inherits the erroneous BMI formula from prior tutor context. The model does not detect or correct wrong prior hints.

2. **Understanding Check Meta-Template**: 6/20 responses (all `follow_up_questions` + `diagnostic_questions` items) use a generic meta-question as the understanding check ("What is a good question to test deep understanding") rather than a concept-specific comprehension probe. This is a consistent template-level weakness scoring 2 vs 3.

3. **Document Grounding (1.15/4)**: Still the lowest-scoring dimension. Retrieved evidence is consistently present in context but almost never explicitly cited. The best grounding achieved is implicit operationalization of formulas (MPC, Assets=Liabilities+Equity). No response cites the source document by name.

4. **Prerequisite Scaffolding**: When weak prerequisites are identified, the response names the gap but rarely proposes a structured bridging activity for the learner. Average prerequisite_detection=2.5 — functional identification, weak scaffolding.

---

## Schema Validity

| Metric | Count |
|---|---|
| Total responses | 20 |
| Schema valid | 20 |
| Schema invalid | 0 |
| task=null in raw_response body | 1 (general_academic_practice_questions_008) |

---

## Key Highlights

**Best response**: `mathematics_misconception_detection_012` — the `f(x)=|x|` counterexample achieves top scores across factual_correctness, misconception_detection, strategy_switching, context_retention, understanding_check_quality, and memory_candidate_quality (all 4/4).

**Most dramatic improvement**: Teaching Strategy Selection (1.21 → 2.05 → 3.70). v0.3 demonstrates that injecting `task` and `instruction` into the ChatML prompt, combined with task-specific training data, resolves the concrete_example override collapse observed in v0.2.

**Elimination of caloric bleed**: Zero cross-domain caloric formulas appear in non-nutrition concept explanations in v0.3. This is the cleanest improvement relative to v0.2.

---

```
V03_SEMANTIC_EVALUATION = PASS
```
