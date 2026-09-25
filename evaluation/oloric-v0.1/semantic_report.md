# Oloric v0.1 — Post-Training Semantic Evaluation Report

**Date**: September 25, 2026
**Model**: Oloric v0.1 (Qwen3-4B-Instruct-2507 + QLoRA adapter, `./checkpoints/final_model` on the L4 training host)
**Benchmark**: `evaluation/benchmark.jsonl` (20 held-out examples, structured mode)
**Rubric**: `evaluation/baseline/semantic_rubric.md` (11 dimensions, 0–4 scale, independent scores, no weighted aggregate)

> **Score-validity statement**: The automated `3.00` values produced by `evaluation_report_1790356740.json` are **INVALID**. They originated from a placeholder in `src/oloric/evaluation.py::_score_dimension()` that returned a constant `3.0` for every dimension of every example. No automated or human semantic evaluation took place in that run. Those numbers were never used as evidence of model quality, and the placeholder has been removed: the automated harness now performs structural checks only and refuses to emit fabricated scores (`NotImplementedError` if any code path attempts to score).

---

## 1. Evaluation methodology

- **Rubric**: the baseline semantic rubric is applied unchanged — same 11 dimensions, same 0–4 anchors, same N/A rules (`strategy_switching` only when prior tutor attempts exist; `misconception_detection` only when a misconception is present; `diagnostic_question_quality` only when diagnostic questions are asked; `memory_candidate_quality` only when a memory candidate exists).
- **Evidence-based and response-specific**: every numeric score must be justified by concrete observations quoting or pinpointing behavior in that response. No defaulting to 3, no identical scores without genuinely identical evidence, no inference from training loss.
- **Like-for-like alignment**: baseline structured-mode responses (`evaluation/baseline/responses.jsonl`, `mode=structured`) and their existing semantic scores (`evaluation/baseline/scores_semantic.jsonl`) are used **as-is** — verified to cover exactly the 20 benchmark IDs, in the same order, and not recomputed.
- **Schema-invalid handling**: `general_academic_practice_questions_008` is preserved schema-invalid. Its raw response, validation failure, failed field, and structural status are recorded (`failure_cases.md`); all 11 semantic dimensions are explicitly **N/A** — the output is not silently repaired and no scores are invented.
- **No weighted overall**: as in the baseline evaluation, no overall weighted score is calculated. The rubric's dimensions are scored independently and the config weights are not used to collapse them into a single winner.
- **Tooling change**: `scripts/evaluate.py` now always writes `evaluation/<run>/responses.jsonl` (streamed per-example, including `raw_generated_text`), so every future run preserves the exact raw outputs needed for manual scoring; `src/oloric/evaluation.py` no longer produces any automated semantic scores. A `run_config.json` with the exact generation settings, benchmark hash, GPU info, and timings is saved alongside the responses.

## 2. Structural results

| Metric | Result |
|---|---|
| Generations completed | **20/20** |
| Schema-valid | **19/20** |
| Schema-invalid | **1/20** (`general_academic_practice_questions_008`, rate 0.05) |
| Failure mode | Unexpected root-level `question` key (`extra_forbidden` under `OloricModelOutput.Config.extra="forbid"`); `understanding_check.question` omitted from the nested object |
| Prior baseline defect (`diagnosis.severity` missing) | **RESOLVED** — `diagnosis.severity` present in all Oloric responses, including the invalid one (`"low"`) |

These structural results are the only trustworthy quantitative outputs of the original run. Baseline structural parity: the Qwen baseline structured run also had 19/20 valid, with a different failure (missing `diagnosis.severity` on `nutrition_food_science_hint_based_teaching_007`).

## 3. Oloric semantic evaluation

**Status: PENDING — 1 of 20 scored, 19 awaiting raw responses.**

- `general_academic_practice_questions_008`: scored as **all N/A** (schema-invalid; see Section 6). Recorded in `evaluation/oloric-v0.1/scores_semantic.jsonl` with full metadata and no numeric scores.
- The other 19 entries exist in `scores_semantic.jsonl` with `status: PENDING_RE_RUN` and `"PENDING"` in every dimension — **no placeholder 3.0s, no fabricated scores**.
- Reason: the original run never wrote `responses.jsonl`. `scripts/evaluate.py` persisted only the schema-failure raw text inside the report; the 19 valid raw outputs were lost when the run directory was not transferred from the GPU host. Only 1 of 20 raw Oloric outputs exists anywhere in the repository.
- Scoring will be completed immediately after the re-run (command below) produces `evaluation/oloric-v0.1/responses.jsonl`, using the identical rubric and evidence standards as the baseline evaluation.

## 4. Baseline-vs-Oloric dimension comparison

The aligned comparison table lives in `evaluation/oloric-v0.1/semantic_comparison.md`. It is already fully aligned on IDs and populated on the baseline side (all 20 rows, 11 dimensions each, with condensed evidence). The Oloric and Δ columns are intentionally blank pending real scores.

Baseline reference (structured mode, from the existing semantic evaluation — not recomputed): strongest dimensions are `factual_correctness` (mean 3.85) and `simplicity` (3.70); weakest are `context_retention` (1.45), `document_grounding` (1.45), and `prerequisite_detection` (1.71); the core pedagogical dimensions `teaching_strategy_selection` (2.55), `strategy_switching` (1.67 over 3 applicable), and `diagnostic_question_quality` (1.80 over 5 applicable) are where tutoring behavior matters most.

No dimension-by-dimension improvement/regression claims are made until Oloric responses are scored. Comparison will be reported per-dimension (e.g. "TSS: baseline 2.55 → Oloric X (Δ +Y)"), never as a single weighted winner.

## 5. Concrete behavioral changes

Only behavioral changes that are **structural and already verifiable** are listed. All semantic/behavioral claims beyond these await scoring.

1. **`diagnosis.severity` emission acquired** (20/20 present, vs. baseline 19/20 with one hard failure on this exact field). This was the baseline structured-mode's only schema failure and is now uniformly correct.
2. **New failure mode introduced**: root-level `question` key leak on a `practice_questions` scenario (`general_academic_practice_questions_008`). The model mirrored the category name ("practice questions") by emitting a `question` key outside `understanding_check`. Baseline never exhibited this mode.
3. **Failure-mode substitution on the same ID**: for `general_academic_practice_questions_008`, the baseline structured response was schema-valid but semantically disobedient (`action=explain` instead of practice problems); Oloric produced practice-problem content but violated the schema. Content intent improved on that item; structural compliance regressed on it.

Everything else — strategy obedience, hint behavior, numerical examples, context use, memory quality — is **not yet characterized** and will be filled in from the re-run responses.

## 6. Remaining failure cases

Documented in `evaluation/oloric-v0.1/failure_cases.md`:

- **Confirmed (structural)**: `general_academic_practice_questions_008` — root-level `question` key, preserved verbatim with validation error, failed field, and structural status. All semantic dimensions explicitly N/A.
- **Pending (behavioral)**: the 8 baseline failure modes carried as explicit checks for the re-run scoring (lecture override, answering own questions, numerical blindness, ignoring erroneous priors, recall-only checks, prerequisite blindness, over-broad memory anchors, plus any novel failure modes). No behavioral failure is claimed or ruled out yet.

## 7. Examples of improvement

**Confirmed so far (structural only):**
- `diagnosis.severity` now present across all 20 responses — the exact defect that failed the baseline's structured run is resolved, including in the one schema-invalid output.

**Expected candidates, to be confirmed by scoring (not yet evidence-backed):** strategy/action alignment, memory-candidate quality, understanding-check depth. These are listed only as the checks the scoring pass will perform first, because they were the baseline's weakest areas.

## 8. Examples of regression

**Confirmed so far (structural only):**
- `general_academic_practice_questions_008` regressed from schema-valid (baseline) to schema-invalid (Oloric) via the root-level `question` leak — while simultaneously improving the *content* on that item from disobedient `explain` to actual practice problems. Recorded as both an improvement candidate (content) and a regression (structure).

**Not yet claimable:** any semantic regression. The comparison will populate this section with specific IDs, dimensions, and evidence once responses are scored.

## 9. Limitations

1. **Missing raw responses (primary limitation).** The 19 valid Oloric responses were never persisted: `scripts/evaluate.py` only saved schema-failure text, and the GPU run directory was not transferred. Consequently 19/20 semantic evaluations are pending. Fixed for future runs: `evaluate.py` now streams every raw response (with `raw_generated_text`) to `responses.jsonl` during the run.
2. **Re-run required for semantic scores.** Until `evaluation/oloric-v0.1/responses.jsonl` exists, no dimension-by-dimension comparison is possible. The automated 3.00s are invalid and are not used as a substitute.
3. **Sampling variance.** The original run's effective generation settings (max_new_tokens 512, temperature 0.7) deviated from the baseline run; they have since been **aligned 1:1 with the baseline** (`evaluation/baseline/config.json`): max_new_tokens=1024, temperature=0.3, top_p=0.9, do_sample=True, **no random seed** (verified: the baseline script set no seed anywhere). Deterministic generation was deliberately NOT adopted because it would *not* match the baseline configuration — so a re-run remains a new stochastic sample at the baseline's operating point, and the single schema failure may or may not reproduce.
4. **Single reviewer, single run, n=20.** All scores (baseline and Oloric) come from one agent reviewer on one 20-example sample; dimension means over ≤3 applicable items (e.g. `strategy_switching`) are especially fragile. Differences of ±1 on a dimension with n≤5 should be treated as noise, not signal.
5. **Benchmark contamination caveat.** Several benchmark `target` fields share template text with the training data (verified: the `general_academic_practice_questions_008` target appears nearly verbatim in the generated training corpora). High similarity between model output and benchmark targets should be interpreted cautiously; the schema leak on that very item suggests the model is echoing template structure.
6. **No weighted overall score** is calculated in this report or the comparison, consistent with the rubric's independent-dimensions design.

---

## Re-run command (GPU host)

```bash
python scripts/evaluate.py \
    --model-path ./checkpoints/oloric-v0.1/final_model \
    --benchmark-file ./evaluation/benchmark.jsonl \
    --output-dir ./evaluation/oloric-v0.1
```

(These are also the new script defaults.) The run writes `evaluation/oloric-v0.1/responses.jsonl` incrementally after every completed generation (all 20 raw responses, each with `raw_generated_text`), a `run_config.json` recording the exact settings, and a structural report with no fabricated scores. It prints a final verification summary (records written, schema-valid/invalid counts, timings, generation settings). Transfer `responses.jsonl` back and the 19 pending semantic evaluations will be completed against the rubric with evidence.

---

**STATUS**: structural evaluation complete; semantic evaluation 1/20 (schema-invalid item, all N/A) + 19 pending re-run.

POST_TRAINING_SEMANTIC_EVALUATION = PENDING_RE_RUN
PLACEHOLDER_SCORES_USED = NO
