# Oloric v0.1 Failure Cases

**Run**: `evaluation_report_1790356740.json` (20 benchmark examples, post-training)
**Status**: Structural evaluation only. Semantic scoring of the 19 valid responses is pending the re-run that exports raw responses (see `semantic_report.md`, Section 9: Limitations).

---

## Case 1 — Schema failure: `general_academic_practice_questions_008`

### Recorded fields (per protocol)

| Field | Value |
|---|---|
| **benchmark_id** | `general_academic_practice_questions_008` |
| **structural status** | `schema_invalid` (counted in `num_schema_failures = 1`, rate 0.05) |
| **failed field** | `question` — unexpected root-level key; `understanding_check.question` omitted from the nested object |
| **validation failure** | `1 validation error for OloricModelOutput`\n`question`\n`  Extra inputs are not permitted [type=extra_forbidden, input_value='What is the key skill dem... and rational thought.', input_type=str]` |
| **schema rule violated** | `OloricModelOutput.Config.extra = "forbid"` |
| **raw output modified?** | **NO** — preserved verbatim below and in the evaluation report; not repaired |

### Raw response (verbatim, unmodified)

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

### Structural analysis

The model emitted `question` at the JSON root (likely triggered by the scenario category `practice_questions`) instead of nesting it inside `understanding_check`. The nested `understanding_check` object contains only `required` and `expected_answer`; its `question` field is missing. Under `extra = "forbid"`, the root-level key is a hard contract violation, so the entire output is `schema_invalid`.

The output was **not repaired** for evaluation. Semantic scores for this item are recorded as **N/A** in `scores_semantic.jsonl` rather than invented, because a structurally invalid response cannot be judged like-for-like against schema-valid responses without silently fixing it.

### Observational remarks (unscored, for the record)

- The readable `response` text does contain 3 practice problems consistent with the `practice_questions` task — the *content* intent is reasonable even though the *structure* failed.
- `diagnosis.severity` is present (`"low"`) — the baseline defect (missing `diagnosis.severity`) remains resolved.
- This is a **new, different failure mode** from the baseline's structured-mode failure (which was missing `diagnosis.severity` on `nutrition_food_science_hint_based_teaching_007`).

### Baseline cross-reference (same benchmark_id)

For contrast: the **baseline Qwen structured response** for this same ID was schema-valid but semantically weak — it used `action: "explain"` instead of producing practice problems, scoring `teaching_strategy_selection = 2` and `strategy_switching = 2` in `evaluation/baseline/scores_semantic.jsonl`. The baseline run's *general*-mode response for this ID did produce practice problems (scored `teaching_strategy_selection = 3`). Oloric v0.1 regenerated practice-problem content but leaked the `question` key to the root, trading a semantic defect for a structural one on this item.

### Likely cause and remediation direction (for the next training iteration — no retraining performed now)

1. The training data's `target` for practice-questions-style categories contains a top-level `question` inside `understanding_check` only; some generated outputs mirror the *category name* ("practice **questions**") by emitting a `question` key. A targeted fine-tuning pass or stricter constrained decoding (grammar/JSON-schema-guided generation) would eliminate the leak.
2. Constrained decoding with the Pydantic schema (e.g. `outlines`/`lm-format-enforcer`) would make this failure class structurally impossible without touching the dataset or benchmark.

---

## Cases 2+ — Behavioral failures (pending re-run)

The 19 schema-valid responses have **not yet been semantically scored** because their raw text was not persisted by the original run (`scripts/evaluate.py` did not export `responses.jsonl`; only the schema-failure text survives inside the report JSON).

After the GPU re-run produces `evaluation/oloric-v0.1/responses.jsonl`, this section will be populated with evidence-based behavioral failure cases, checked against the baseline's known failure modes (see `evaluation/baseline/failure_cases.md`):

1. **Lecture override** — does Oloric still default to full explanations for `hint_based_teaching` / `follow_up_questions` / `practice_questions` tasks?
2. **Answering its own questions** — does Oloric still provide inline answers to diagnostic/follow-up questions?
3. **Numerical blindness** — does Oloric produce actual numbers for `numerical_example` tasks (baseline scored 1–2 on the affected items)?
4. **Ignoring erroneous priors** — does Oloric acknowledge/correct the wrong prior hint (`nutrition_food_science_hint_based_teaching_007`)?
5. **Recall-only understanding checks** — are checks application/transfer questions or rote recall?
6. **Prerequisite blindness** — are flagged `weak_prerequisites` probed or scaffolded?
7. **Memory anchor precision** — are `anchor_concept` values specific (e.g. "Dietary Fiber") or overly broad (e.g. "Nutrition_food_science")?
8. **New failure modes** — anything the baseline did not exhibit (e.g. root-level key leaks, as in Case 1).

No behavioral failure is claimed or ruled out until the raw responses are scored.
