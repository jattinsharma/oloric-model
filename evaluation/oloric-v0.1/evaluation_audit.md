# OLORIC v0.1 Post-Training Evaluation Audit

**Audit Date**: September 25, 2026  
**Subject**: Inspection of Oloric v0.1 20-Example Benchmark Evaluation (`evaluation_report_1790356740.json`)  
**Status**: COMPLETE  

---

## Executive Summary

The evaluation run of Oloric v0.1 across the 20 held-out benchmark examples completed without infrastructure crashes, producing `evaluation_report_1790356740.json`. However, the final report reported:
- **Overall score**: `3.00`
- **Every one of the 11 dimensions**: `3.00`
- **Every dimension range**: `3.00 - 3.00`

This audit investigated the source of these numbers, traced the code execution path, analyzed the single observed schema failure (`general_academic_practice_questions_008`), and reviewed the structural versus semantic evaluation framework.

**Key Finding**: The reported `3.00` score is **100% an unreplaced code placeholder**, hardcoded directly in `src/oloric/evaluation.py`. No automated or human semantic evaluation took place during this run. The reported semantic numbers are therefore **INVALID** and must not be used to assess model capability.

---

## TASK 1 — Inspection of the Actual Report

Inspection of `evaluation/oloric-v0.1/evaluation_report_1790356740.json` reveals:

1. **Per-Example Scores**:
   - **Not present.** The report generation method (`generate_report` in `src/oloric/evaluation.py`) only computes aggregate summary statistics across valid examples. It discards individual per-example scores.
2. **Per-Dimension Scores**:
   - All 11 dimensions (`factual_correctness`, `document_grounding`, `simplicity`, `teaching_strategy_selection`, `strategy_switching`, `misconception_detection`, `prerequisite_detection`, `context_retention`, `diagnostic_question_quality`, `understanding_check_quality`, `memory_candidate_quality`) report identically:
     ```json
     {
       "mean": 3.0,
       "min": 3.0,
       "max": 3.0
     }
     ```
3. **Score Distribution**:
   - All scores across every valid example are literally `3.00`. The variance is exactly `0.0`.
4. **Evidence and Notes**:
   - **None.** The report contains zero qualitative evidence, zero rationale, zero quotes from model outputs, and zero notes explaining why any score was awarded.
5. **Schema Failures Representation**:
   - Schema failures are captured in the `schema_failures` list:
     - `num_examples`: 20
     - `num_schema_valid`: 19
     - `num_schema_failures`: 1 (`schema_failure_rate`: 0.05)
     - The single failure recorded is `general_academic_practice_questions_008`.

---

## TASK 2 — Trace of the Scoring Implementation

Tracing the codebase reveals exactly where and why the value `3.0` was generated:

### 1. Hardcoded Placeholder in `src/oloric/evaluation.py`
In `src/oloric/evaluation.py`, lines 136–154:

```python
def _score_dimension(self, dimension_name: str, context: OloricModelInput,
                    model_output: OloricModelOutput, 
                    expected_output: OloricModelOutput) -> float:
    """
    Score a specific evaluation dimension.
    
    Args:
        dimension_name: Name of dimension to score
        context: Input context
        model_output: Model's output
        expected_output: Expected output
        
    Returns:
        Score from 0.0 to 4.0
    """
    # Placeholder implementation - returns fixed scores for now
    # In a real implementation, each dimension would have specific scoring logic
    return 3.0  # Placeholder score
```

### 2. Weighted Aggregation in `_calculate_overall_score`
In `src/oloric/evaluation.py`, lines 155–177:
```python
overall_score = sum(
    score * weights[name] for name, score in dimension_scores.items()
)
```
Because every dimension score returned by `_score_dimension` is `3.0`, any convex combination $\sum w_i \cdot 3.0$ where $\sum w_i = 1.0$ evaluates mathematically to `3.00`.

### 3. Report Summary Aggregation in `generate_report`
In `src/oloric/evaluation.py`, lines 240–265:
For all 19 valid examples, `dim_values = [3.0, 3.0, ... 3.0]`.  
- `mean`: $\frac{19 \times 3.0}{19} = 3.00$
- `min`: `3.00`
- `max`: `3.00`

### 4. Determination
- **Determination**: **B. Producing a placeholder/default score.**
- The evaluation engine did not run any semantic judge, rule-based heuristic, or rubric evaluator. It simply executed the placeholder `return 3.0`.

---

## TASK 3 — Prohibition on Faking Semantic Evaluation

The 11 evaluation dimensions:
1. `factual_correctness` (0–4)
2. `simplicity` (0–4)
3. `teaching_strategy_selection` (0–4)
4. `strategy_switching` (0–4)
5. `misconception_detection` (0–4)
6. `prerequisite_detection` (0–4)
7. `context_retention` (0–4)
8. `diagnostic_question_quality` (0–4)
9. `understanding_check_quality` (0–4)
10. `memory_candidate_quality` (0–4)
11. `document_grounding` (0–4)

As explicitly codified in `evaluation/baseline/semantic_rubric.md`:
> *"Important: These dimensions require human judgment. They are NOT deterministically computable. Automated structural checks (schema validity, field presence) are handled separately by score_baseline_automated.py."*

- **Automated evaluation** in Oloric is strictly capable of structural scoring: JSON validity, Pydantic schema compliance, enum restrictions, and field presence.
- **Semantic evaluation** cannot be simulated with simplistic heuristics or hardcoded defaults. Faking semantic evaluation distorts training progress and risks deploying ungrounded models.
- The earlier manual baseline evaluation (`evaluation/baseline/scores_semantic.jsonl`) evaluated all 40 baseline responses (20 general + 20 structured) response-by-response with detailed per-item critique and evidence notes. Post-training evaluation must adhere to this exact standard.

---

## TASK 4 — Schema Failure Analysis: `general_academic_practice_questions_008`

The single schema failure in the 20-example evaluation was investigated in detail:

### 1. Example Metadata
- **ID**: `general_academic_practice_questions_008`
- **Domain**: `general_academic`
- **Category**: `practice_questions`
- **Instruction**: `"Provide practice problems to reinforce understanding of Effective Communication."`
- **Context Task**: `resolve_confusion`

### 2. Expected Output Schema
The contract is defined by `OloricModelOutput` in `src/oloric/schemas/model_output.py`:
```python
class OloricModelOutput(BaseModel):
    task: Optional[str]
    action: str
    strategy: str
    difficulty: str
    response: str
    understanding_check: UnderstandingCheck  # contains required, question, expected_answer
    diagnosis: Diagnosis                    # contains confusion_type, severity, misconception_addressed
    memory: Memory                          # contains candidate, title, content, anchor_concept, memory_type, confidence

    class Config:
        extra = "forbid"
```

### 3. Raw Generated Model Response
```json
{
  "action": "explain",
  "strategy": "practice",
  "difficulty": "beginner",
  "response": "Let's practice Effective Communication with some exercises:\n\nProblem 1: If a speaker uses clear language, organized structure, and engaging delivery, they communicate effectively.\n\nProblem 2: Explain how Effective Communication relates to basic literacy.\n\nProblem 3: Describe a real-world application of Effective Communication in general_academic.",
  "question": "What is the key skill demonstrated in solving these Effective Communication problems?",
  "understanding_check": {
    "required": true,
    "expected_answer": "The key skill is applying Effective Communication to solve problems because it underpins rigorous academic work and rational thought."
  },
  "diagnosis": {
    "confusion_type": "conceptual",
    "severity": "low",
    "misconception_addressed": null
  },
  "memory": {
    "candidate": true,
    "title": "Practice problems for Effective Communication",
    "content": "Key practice types for Effective Communication: calculation, explanation, application.",
    "anchor_concept": "Effective Communication",
    "memory_type": "procedure",
    "confidence": 0.66
  }
}
```

### 4. Exact Invalid Field and Path
- **Path**: `question` (top-level / root dictionary key).
- **Validation Error**:
  ```text
  1 validation error for OloricModelOutput
  question
    Extra inputs are not permitted [type=extra_forbidden, input_value='What is the key skill dem... and rational thought.', input_type=str]
  ```

### 5. Level Discrepancy
- **Did the model produce `question` at the wrong level?**  
  **YES.** Because the scenario category was `practice_questions`, the model emitted `"question"` as a top-level JSON key alongside `"response"`, `"action"`, and `"strategy"`. It omitted `"question"` from inside `"understanding_check"`, leaving only `"required": true` and `"expected_answer": "..."` in the nested object.

### 6. Classification
- **Should the response be counted as schema_invalid?**  
  **YES.** Under `extra = "forbid"`, any unexpected root-level field is a structural contract violation. Silently repairing the JSON or dropping the extra key during evaluation would mask a real architectural error in model generation. The evaluation framework correctly flagged this example with `schema_valid: false`.

---

## TASK 5 — Valid Post-Training Evaluation Path

To perform an honest, trustworthy comparison between **Qwen Baseline** and **Oloric v0.1** on the identical 20 benchmark examples:

### Step 1: Automated Structural Scoring
1. Save all 20 raw model outputs to `evaluation/oloric-v0.1/responses.jsonl`.
2. Run `scripts/score_baseline_automated.py`:
   ```bash
   python scripts/score_baseline_automated.py \
     --responses evaluation/oloric-v0.1/responses.jsonl \
     --output-dir evaluation/oloric-v0.1
   ```
   This generates:
   - `evaluation/oloric-v0.1/scores_automated.jsonl`
   - `evaluation/oloric-v0.1/scores_automated_summary.json`

### Step 2: Manual Semantic Scoring Workflow
1. Create `evaluation/oloric-v0.1/scores_semantic.jsonl` with 20 entries mirroring the baseline format:
   ```json
   {
     "id": "<example_id>",
     "mode": "structured",
     "reviewer": "evaluator",
     "factual_correctness": <0-4>,
     "simplicity": <0-4>,
     "teaching_strategy_selection": <0-4>,
     "strategy_switching": <0-4 or N/A>,
     "misconception_detection": <0-4 or N/A>,
     "prerequisite_detection": <0-4 or N/A>,
     "context_retention": <0-4>,
     "diagnostic_question_quality": <0-4 or N/A>,
     "understanding_check_quality": <0-4>,
     "memory_candidate_quality": <0-4 or N/A>,
     "document_grounding": <0-4>,
     "notes": "<Specific evidence and pedagogical rationale>"
   }
   ```
2. Score each example against `evaluation/baseline/semantic_rubric.md`.
3. Generate the side-by-side comparative report (`evaluation/oloric-v0.1/semantic_comparison.md`).

---

## TASK 6 — Structural Results Summary

### Baseline (Qwen Structured) vs. Oloric v0.1

| Metric | Qwen Baseline (Structured) | Oloric v0.1 (Post-Training) | Status |
| :--- | :--- | :--- | :--- |
| **Total Benchmark Examples** | 20 | 20 | Identical |
| **Completed Generations** | 20 / 20 (100%) | 20 / 20 (100%) | Parity |
| **Schema Valid Responses** | 19 / 20 (95.0%) | 19 / 20 (95.0%) | Parity |
| **Schema Violations** | 1 / 20 (5.0%) | 1 / 20 (5.0%) | Parity |
| **Specific Failure Mode** | Missing required `diagnosis.severity` (`nutrition_007`) | Root-level extra `question` key (`general_academic_008`) | Different failure |
| **Resolved Defect** | N/A | `diagnosis.severity` is now present across all 20 examples | RESOLVED |
| **New Defect** | N/A | Practice questions root-level key leak | IDENTIFIED |

---

## What Remains to Be Evaluated

1. **20-Example Semantic Scoring**: Manual rubric-based scoring across all 11 dimensions using `evaluation/baseline/semantic_rubric.md`.
2. **Strategy Alignment**: Verification of whether Oloric v0.1 selected appropriate teaching strategies (`explain`, `analogy`, `hint`, `practice`) matching learner mastery and misconceptions.
3. **Prerequisite & Misconception Handling**: Assessment of whether detected gaps led to appropriate prerequisite remediation.
4. **Memory Quality**: Evaluation of whether memory candidates accurately isolate anchor concepts and actionable summaries.

---

## Recommended Next Steps

1. **Remove Deceptive Placeholder**: Update `src/oloric/evaluation.py` so `_score_dimension` returns `None` or raises an explicit notice that semantic scoring requires the semantic evaluation rubric, preventing automated reports from presenting artificial `3.00` scores.
2. **Export Raw Model Responses**: Ensure `scripts/evaluate.py` writes `responses.jsonl` containing all 20 full model outputs.
3. **Conduct Human / Expert Semantic Review**: Populate `evaluation/oloric-v0.1/scores_semantic.jsonl` with genuine scores (0–4) and evidence notes.
4. **Publish Comparative Evaluation**: Output final Gate 1 comparison metrics contrasting Qwen Base vs. Oloric v0.1.

---

POST_TRAINING_SCORE_VALIDITY = INVALID
