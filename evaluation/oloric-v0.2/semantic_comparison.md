# Oloric v0.2 — Semantic Comparison: Baseline vs v0.1 vs v0.2

**Rubric**: `evaluation/baseline/semantic_rubric.md` (0–4 scale, independently per dimension)  
**Baseline mode**: General (unstructured Qwen3-4B-Instruct-2507)  
**v0.1 / v0.2 mode**: Structured (Qwen3-4B + trained LoRA adapter)

> [!IMPORTANT]
> There is no single weighted overall score. Comparison is dimension-by-dimension only, as required. All three evaluations used identical rubric and reviewer methodology.

---

## Dimension-by-Dimension Comparison

### D1: Factual Correctness

| ID | Baseline | v0.1 | v0.2 | Δ (v1→v2) | Δ (base→v2) |
|----|:--------:|:----:|:----:|:---------:|:----------:|
| nutrition_food_science_hint_based_teaching_007 | 4 | 0 | 1 | +1 | -3 |
| general_academic_concrete_example_040 | 4 | 0 | 3 | +3 | -1 |
| economics_diagnostic_questions_013 | 4 | 0 | 2 | +2 | -2 |
| nutrition_food_science_follow_up_questions_016 | 4 | 0 | 2 | +2 | -2 |
| mathematics_follow_up_questions_035 | 3 | 0 | 2 | +2 | -1 |
| general_academic_simple_explanation_029 | 4 | 0 | 3 | +3 | -1 |
| economics_numerical_example_029 | 4 | 0 | 3 | +3 | -1 |
| nutrition_food_science_diagnostic_questions_009 | 4 | 0 | 3 | +3 | -1 |
| nutrition_food_science_numerical_example_003 | 3 | 0 | 1 | +1 | -2 |
| mathematics_misconception_detection_012 | 4 | 0 | 2 | +2 | -2 |
| science_numerical_example_030 | 4 | 0 | 2 | +2 | -2 |
| nutrition_food_science_numerical_example_001 | 3 | 0 | 2 | +2 | -1 |
| general_academic_misconception_detection_008 | 4 | 0 | 2 | +2 | -2 |
| science_diagnostic_questions_013 | 4 | 0 | 2 | +2 | -2 |
| science_follow_up_questions_020 | 4 | 0 | 3 | +3 | -1 |
| accountancy_diagnostic_questions_003 | 4 | 0 | 3 | +3 | -1 |
| economics_follow_up_questions_009 | 4 | 0 | 3 | +3 | -1 |
| general_academic_practice_questions_008 | 4 | N/A | 2 | — | -2 |
| accountancy_prerequisite_detection_014 | 4 | 0 | 2 | +2 | -2 |
| nutrition_food_science_follow_up_questions_035 | 4 | 0 | 3 | +3 | -1 |
| **Mean** | **3.85** | **0.05** | **2.20** | **+2.15** | **-1.65** |

**Finding**: v0.2 eliminates the zero-factual-content failure of v0.1. All responses contain real facts. However, v0.2 still trails baseline by 1.65 points on average — primarily due to cross-domain example bleed causing false attributions.

---

### D2: Simplicity

| ID | Baseline | v0.1 | v0.2 | Δ (v1→v2) |
|----|:--------:|:----:|:----:|:---------:|
| nutrition_food_science_hint_based_teaching_007 | 3 | 1 | 2 | +1 |
| general_academic_concrete_example_040 | 4 | 1 | 3 | +2 |
| economics_diagnostic_questions_013 | 3 | 1 | 3 | +2 |
| nutrition_food_science_follow_up_questions_016 | 3 | 1 | 2 | +1 |
| mathematics_follow_up_questions_035 | 3 | 1 | 2 | +1 |
| general_academic_simple_explanation_029 | 4 | 1 | 3 | +2 |
| economics_numerical_example_029 | 4 | 1 | 3 | +2 |
| nutrition_food_science_diagnostic_questions_009 | 4 | 1 | 3 | +2 |
| nutrition_food_science_numerical_example_003 | 4 | 1 | 2 | +1 |
| mathematics_misconception_detection_012 | 3 | 1 | 2 | +1 |
| science_numerical_example_030 | 4 | 1 | 3 | +2 |
| nutrition_food_science_numerical_example_001 | 4 | 1 | 3 | +2 |
| general_academic_misconception_detection_008 | 3 | 1 | 2 | +1 |
| science_diagnostic_questions_013 | 4 | 1 | 3 | +2 |
| science_follow_up_questions_020 | 4 | 1 | 3 | +2 |
| accountancy_diagnostic_questions_003 | 4 | 1 | 3 | +2 |
| economics_follow_up_questions_009 | 3 | 1 | 3 | +2 |
| general_academic_practice_questions_008 | 3 | N/A | 3 | — |
| accountancy_prerequisite_detection_014 | 4 | 1 | 3 | +2 |
| nutrition_food_science_follow_up_questions_035 | 3 | 1 | 3 | +2 |
| **Mean** | **3.60** | **1.00** | **2.70** | **+1.70** |

**Finding**: Simplicity improves markedly. v0.2 responses are consistently clear, beginner-accessible prose. Still below baseline because some responses use terms mismatched to concept (e.g., calculus terms for matrix operations).

---

### D3: Teaching Strategy Selection

| ID | Baseline | v0.1 | v0.2 | Δ (v1→v2) |
|----|:--------:|:----:|:----:|:---------:|
| nutrition_food_science_hint_based_teaching_007 | 1 | 1 | 2 | +1 |
| general_academic_concrete_example_040 | 4 | 2 | 3 | +1 |
| economics_diagnostic_questions_013 | 3 | 1 | 3 | +2 |
| nutrition_food_science_follow_up_questions_016 | 2 | 1 | 1 | 0 |
| mathematics_follow_up_questions_035 | 2 | 1 | 1 | 0 |
| general_academic_simple_explanation_029 | 4 | 3 | 3 | 0 |
| economics_numerical_example_029 | 2 | 1 | 3 | +2 |
| nutrition_food_science_diagnostic_questions_009 | 3 | 1 | 1 | 0 |
| nutrition_food_science_numerical_example_003 | 3 | 1 | 2 | +1 |
| mathematics_misconception_detection_012 | 3 | 1 | 3 | +2 |
| science_numerical_example_030 | 4 | 1 | 2 | +1 |
| nutrition_food_science_numerical_example_001 | 3 | 1 | 2 | +1 |
| general_academic_misconception_detection_008 | 3 | 2 | 3 | +1 |
| science_diagnostic_questions_013 | 4 | 1 | 1 | 0 |
| science_follow_up_questions_020 | 2 | 1 | 3 | +2 |
| accountancy_diagnostic_questions_003 | 3 | 1 | 1 | 0 |
| economics_follow_up_questions_009 | 2 | 1 | 1 | 0 |
| general_academic_practice_questions_008 | 3 | N/A | 2 | — |
| accountancy_prerequisite_detection_014 | 4 | 1 | 3 | +2 |
| nutrition_food_science_follow_up_questions_035 | 2 | 1 | 1 | 0 |
| **Mean** | **2.90** | **1.25** | **2.05** | **+0.80** |

**Finding**: Improvement is real (+0.80 vs v0.1) but task-following remains the primary gap. The model systematically selects `concrete_example` even when the task demands `follow_up_questions` or `diagnostic_questions`. 7 of 8 question-generating tasks still fail.

---

### D4: Strategy Switching (3 applicable examples)

| ID | Baseline | v0.1 | v0.2 | Δ (v1→v2) |
|----|:--------:|:----:|:----:|:---------:|
| nutrition_food_science_hint_based_teaching_007 | 1 | 1 | 2 | +1 |
| mathematics_misconception_detection_012 | 3 | 1 | 3 | +2 |
| general_academic_misconception_detection_008 | 2 | 2 | 3 | +1 |
| **Mean** | **2.00** | **1.33** | **2.67** | **+1.33** |

**Finding**: Meaningful improvement. v0.2 attempts a real switch in strategy for misconception tasks. The hint-based example still reuses the prior erroneous hint rather than correcting it.

---

### D5: Misconception Detection (applicable examples)

| ID | Baseline | v0.1 | v0.2 | Δ (v1→v2) |
|----|:--------:|:----:|:----:|:---------:|
| mathematics_misconception_detection_012 | 4 | 2 | 3 | +1 |
| general_academic_misconception_detection_008 | 3 | 2 | 3 | +1 |
| general_academic_simple_explanation_029 | N/A | N/A | 1 | — |
| economics_diagnostic_questions_013 | N/A | 1 | N/A | — |
| nutrition_food_science_diagnostic_questions_009 | N/A | 1 | 1 | 0 |
| nutrition_food_science_numerical_example_003 | N/A | 1 | 1 | 0 |
| science_numerical_example_030 | N/A | 1 | 1 | 0 |
| accountancy_diagnostic_questions_003 | N/A | 1 | 1 | 0 |
| economics_follow_up_questions_009 | N/A | N/A | 1 | — |
| nutrition_food_science_follow_up_questions_035 | N/A | 1 | 1 | 0 |
| **Mean (where both scored)** | — | **1.50** | **1.60** | **+0.10** |

**Finding**: For the two genuine misconception tasks, v0.2 correctly targets the real stated misconception (up from partial in v0.1). The `diagnosis.misconception_addressed` field in non-misconception tasks still injects generic claims — this field is not yet discriminative.

---

### D6: Prerequisite Detection (applicable examples)

| ID | Baseline | v0.1 | v0.2 | Δ (v1→v2) |
|----|:--------:|:----:|:----:|:---------:|
| economics_diagnostic_questions_013 | 1 | N/A | 1 | — |
| mathematics_follow_up_questions_035 | 1 | N/A | 1 | — |
| nutrition_food_science_diagnostic_questions_009 | 2 | N/A | 1 | — |
| nutrition_food_science_numerical_example_003 | N/A | N/A | N/A | — |
| science_diagnostic_questions_013 | 2 | N/A | 2 | — |
| science_follow_up_questions_020 | 1 | N/A | 1 | — |
| accountancy_diagnostic_questions_003 | 2 | N/A | 1 | — |
| economics_follow_up_questions_009 | 1 | N/A | 1 | — |
| accountancy_prerequisite_detection_014 | 4 | 1 | 3 | **+2** |
| nutrition_food_science_follow_up_questions_035 | 1 | N/A | 1 | — |
| **Mean (v0.2 applicable)** | **1.67** | **1.00** | **1.67** | **+0.67** |

**Finding**: `accountancy_prerequisite_detection_014` shows the most notable improvement (+2 from v0.1). Most other examples still score 1 — the model does not explicitly check or scaffold weak prerequisites flagged in learner state.

---

### D7: Context Retention

| ID | Baseline | v0.1 | v0.2 | Δ (v1→v2) |
|----|:--------:|:----:|:----:|:---------:|
| nutrition_food_science_hint_based_teaching_007 | 1 | 1 | 1 | 0 |
| general_academic_concrete_example_040 | 2 | 1 | 2 | +1 |
| economics_diagnostic_questions_013 | 2 | 1 | 2 | +1 |
| nutrition_food_science_follow_up_questions_016 | 1 | 2 | 2 | 0 |
| mathematics_follow_up_questions_035 | 1 | 1 | 2 | +1 |
| general_academic_simple_explanation_029 | 2 | 1 | 2 | +1 |
| economics_numerical_example_029 | 1 | 1 | 2 | +1 |
| nutrition_food_science_diagnostic_questions_009 | 2 | 1 | 2 | +1 |
| nutrition_food_science_numerical_example_003 | 1 | 1 | 1 | 0 |
| mathematics_misconception_detection_012 | 3 | 2 | 3 | +1 |
| science_numerical_example_030 | 1 | 1 | 2 | +1 |
| nutrition_food_science_numerical_example_001 | 1 | 1 | 2 | +1 |
| general_academic_misconception_detection_008 | 2 | 2 | 3 | +1 |
| science_diagnostic_questions_013 | 2 | 2 | 2 | 0 |
| science_follow_up_questions_020 | 1 | 1 | 2 | +1 |
| accountancy_diagnostic_questions_003 | 2 | 1 | 2 | +1 |
| economics_follow_up_questions_009 | 1 | 1 | 2 | +1 |
| general_academic_practice_questions_008 | 2 | N/A | 2 | — |
| accountancy_prerequisite_detection_014 | 3 | 2 | 3 | +1 |
| nutrition_food_science_follow_up_questions_035 | 1 | 1 | 2 | +1 |
| **Mean** | **1.65** | **1.25** | **2.10** | **+0.85** |

**Finding**: Consistent +1 improvement across most items. The model now regularly echoes concept, domain, and definition from context. Retrieved evidence and document sections are still not used (ceiling at 2 in most cases).

---

### D8: Diagnostic Question Quality (applicable)

| ID | Baseline | v0.1 | v0.2 | Δ (v1→v2) |
|----|:--------:|:----:|:----:|:---------:|
| economics_diagnostic_questions_013 | 3 | 1 | 1 | 0 |
| nutrition_food_science_diagnostic_questions_009 | 3 | 1 | N/A | — |
| science_follow_up_questions_020 | N/A | N/A | 3 | — |
| **Mean** | **3.00** | **1.00** | **2.00** | **+1.00** |

**Finding**: `science_follow_up_questions_020` is the standout: v0.2 correctly produces 4 substantive diagnostic/follow-up questions targeting prerequisites and prior knowledge. The other diagnostic task IDs still produce zero questions.

---

### D9: Understanding Check Quality

| ID | Baseline | v0.1 | v0.2 | Δ (v1→v2) |
|----|:--------:|:----:|:----:|:---------:|
| nutrition_food_science_hint_based_teaching_007 | 2 | 1 | 2 | +1 |
| general_academic_concrete_example_040 | 2 | 1 | 3 | +2 |
| economics_diagnostic_questions_013 | N/A | 1 | 2 | — |
| nutrition_food_science_follow_up_questions_016 | 3 | 1 | 2 | +1 |
| mathematics_follow_up_questions_035 | 3 | 1 | 2 | +1 |
| general_academic_simple_explanation_029 | 2 | 1 | 3 | +2 |
| economics_numerical_example_029 | 2 | 1 | 3 | +2 |
| nutrition_food_science_diagnostic_questions_009 | N/A | 1 | 3 | — |
| nutrition_food_science_numerical_example_003 | 2 | 1 | 2 | +1 |
| mathematics_misconception_detection_012 | 3 | 1 | 3 | +2 |
| science_numerical_example_030 | 3 | 1 | 3 | +2 |
| nutrition_food_science_numerical_example_001 | 2 | 1 | 3 | +2 |
| general_academic_misconception_detection_008 | 2 | 1 | 2 | +1 |
| science_diagnostic_questions_013 | N/A | 1 | 3 | — |
| science_follow_up_questions_020 | 3 | 1 | N/A | — |
| accountancy_diagnostic_questions_003 | N/A | 1 | 3 | — |
| economics_follow_up_questions_009 | 2 | 1 | 3 | +2 |
| general_academic_practice_questions_008 | 2 | N/A | 2 | — |
| accountancy_prerequisite_detection_014 | 3 | 1 | 3 | +2 |
| nutrition_food_science_follow_up_questions_035 | 3 | 1 | 3 | +2 |
| **Mean** | **2.47** | **1.00** | **2.65** | **+1.65** |

**Finding**: Understanding check quality is the most consistent improvement dimension. v0.2 replaces placeholder check questions with real, substantive recall questions and non-circular expected answers in 16/19 applicable responses.

---

### D10: Memory Candidate Quality (applicable)

| ID | Baseline | v0.1 | v0.2 | Δ (v1→v2) |
|----|:--------:|:----:|:----:|:---------:|
| nutrition_food_science_hint_based_teaching_007 | N/A | 1 | 1 | 0 |
| general_academic_simple_explanation_029 | N/A | N/A | 3 | — |
| general_academic_misconception_detection_008 | N/A | 1 | 3 | +2 |
| nutrition_food_science_numerical_example_001 | N/A | N/A | 2 | — |
| economics_follow_up_questions_009 | N/A | 1 | 2 | +1 |
| **Mean (where both scored)** | — | **1.00** | **2.20** | **+1.20** |

**Finding**: Memory candidates are no longer placeholder. `general_academic_simple_explanation_029` produces a well-anchored, genuinely usable memory candidate (score 3). Memory quality improvement is one of the clearest wins for v0.2.

---

### D11: Document Grounding

| ID | Baseline | v0.1 | v0.2 | Δ (v1→v2) |
|----|:--------:|:----:|:----:|:---------:|
| All 20 (mean) | **1.50** | **0.00** | **0.95** | **+0.95** |
| nutrition_food_science_numerical_example_003 | — | 0 | **0** | 0 |

**Finding**: v0.2 produces a baseline of 1 (not contradicting document context) in 19/20 cases. But no response achieves ≥ 2 — retrieved evidence and document titles/sections are never explicitly incorporated. Baseline (general mode) averaged 1.5 on this dimension, so v0.2 approaches but does not match baseline.

---

## Summary Radar

```
Dimension               Baseline   v0.1    v0.2
────────────────────────────────────────────────
Factual Correctness      3.85      0.05    2.20  ████████▌
Simplicity               3.60      1.00    2.70  ██████████▍
Teaching Strategy        2.90      1.25    2.05  ████████▏
Strategy Switching       2.00      1.33    2.67  ██████████▋
Misconception Detection  3.50      1.50    1.60  ██████▍
Prerequisite Detection   1.80      1.00    1.67  ██████▋
Context Retention        1.65      1.25    2.10  ████████▍
Diagnostic Q Quality     3.00      1.00    2.00  ████████
Understanding Check      2.47      1.00    2.65  ██████████▌
Memory Candidate         N/A       1.00    2.20  ████████▊
Document Grounding       1.50      0.00    0.95  ███▊
```

v0.2 sits consistently between v0.1 and Baseline, with the largest gaps remaining in **Factual Correctness** (−1.65 vs baseline) and **Document Grounding** (−0.55 vs baseline).
