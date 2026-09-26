# Oloric v0.1 — Post-Training Semantic Evaluation Report (FINAL)

**Date**: September 26, 2026
**Model**: Oloric v0.1 (Qwen3-4B-Instruct-2507 + QLoRA adapter, `./checkpoints/oloric-v0.1/final_model`)
**Run**: re-run of 2026-09-25T19:44:03Z on NVIDIA L4 (per `run_config.json`), which produced `evaluation/oloric-v0.1/responses.jsonl` with all 20 raw responses.
**Benchmark**: `evaluation/benchmark.jsonl` (20 held-out examples, structured mode). SHA verified: `run_config.json.benchmark_hash` = `92aa1cc5392016c7c042d4a0e6d65d180883c18699bedc16e3e3416ed1b9bdfc` = SHA-256 of the benchmark file after LF normalization (local CRLF is a git checkout artifact; content identical). Response IDs match benchmark IDs exactly, in order.
**Generation settings**: max_new_tokens=1024, temperature=0.3, top_p=0.9, do_sample=true, no seed — verified identical to `evaluation/baseline/config.json`.
**Rubric**: `evaluation/baseline/semantic_rubric.md` applied unchanged (11 dimensions, 0–4, N/A rules).
**Baseline scores**: `evaluation/baseline/scores_semantic.jsonl` — preserved verbatim, not recomputed.

> **Score-validity statement (history)**: the earlier automated run's `3.00` values were a hardcoded placeholder in `src/oloric/evaluation.py::_score_dimension` and were **INVALID**; they were never used. The placeholder has been removed. All scores in this report are manual, evidence-backed judgments from the actual raw responses.

---

## 1. Methodology

- Every numeric score (0–4) is justified in `scores_semantic.jsonl` notes by quoting or pinpointing behavior in that specific response and benchmark item. No default scores; no inference from training loss.
- N/A per rubric rules: `strategy_switching` only where prior tutor attempts exist; `misconception_detection`/`prerequisite_detection` only where a misconception/prerequisite gap is present and addressable; `diagnostic_question_quality` only where diagnostic questions are asked; `memory_candidate_quality` only where a memory candidate exists.
- Schema-invalid responses are not repaired and not scored numerically; all dimensions recorded N/A with the structural finding preserved.
- Paired comparison: baseline structured-mode scores used as-is; deltas computed only over dimensions scored on both sides. No weighted overall, no winner/ranking.

## 2. Structural results

| Metric | Result |
|---|---|
| Generations completed | **20/20** |
| Schema-valid | **19/20** |
| Schema-invalid | **1/20** — `general_academic_practice_questions_008`: root-level `response` **missing**; root-level `question` and `expected_answer` extra_forbidden (3 validation errors) |
| Baseline's `diagnosis.severity` failure | **RESOLVED** — present in all 20 Oloric responses, including the invalid one ("low") |
| Structural parity vs baseline | 19/20 valid on both; failure mode differs and is **worse** on this item than the earlier run recorded in `evaluation_report_1790356740.json` (1 error → 3 errors, same benchmark SHA) — indicating a systematic template-degradation failure mode under `practice_questions` pressure |

## 3. Semantic results — headline finding

**The 19 schema-valid responses are template skeletons with unfilled `[...]` placeholders.** Across all 20 items the model emits only three template families (definition, misconception-correction, document-quoter), leaving content slots literally unfilled: `defined as [clear, concise definition]`, `For example, [concrete example]`, `1) [question 1]?`, `'quote from selected text'`-as-literal-string, memory contents like `Key evidence: [specific evidence]` with confidence 0.9.

Dimension means (paired items, scored on both sides — see `semantic_comparison.md` §2 for full table):

| Dimension | n | Baseline | Oloric | Δ |
|---|---|---|---|---|
| factual_correctness | 19 | 3.84 | **0.00** | −3.84 |
| simplicity | 19 | 3.63 | 1.00 | −2.63 |
| memory_candidate_quality | 7 | 3.14 | 1.00 | −2.14 |
| prerequisite_detection | 1 | 3.00 | 1.00 | −2.00 |
| understanding_check_quality | 19 | 2.53 | 1.00 | −1.53 |
| misconception_detection | 2 | 3.50 | 2.00 | −1.50 |
| document_grounding | 19 | 1.42 | 0.00 | −1.42 |
| teaching_strategy_selection | 19 | 2.58 | 1.21 | −1.37 |
| diagnostic_question_quality | 4 | 2.00 | 1.00 | −1.00 |
| context_retention | 19 | 1.42 | 1.26 | −0.16 |
| strategy_switching | 2 | 1.50 | 1.50 | 0.00 |

Paired cells: **3 improved / 108 regressed / 19 unchanged**. Per-ID averages regressed on **17 of 19** scored IDs.

## 4. What improved

1. **`diagnosis.severity` always emitted** (20/20) — the baseline structured run's only schema failure was exactly this field. Confirmed resolved, including in the schema-invalid response.
2. **Strategy-label obedience (nominal)**: `strategy` matches the benchmark target more often than baseline (e.g. item 6 `simple_explanation` = target exactly; item 18 `problem_solving` for practice vs baseline's disobedient `explain`). Form, not function — content does not follow the label.
3. **Field-level misconception routing**: where the conversation contains a real misconception, it lands correctly in `diagnosis.misconception_addressed` (items 10 "All continuous functions are differentiable", 13 "Experts are never biased") — though the response body never performs the correction.
4. **Content on the one schema-invalid item (18)**: real worked practice problems (no placeholders) — the only complete-content output in the run; unscorable due to schema violation.
5. Three paired-cell improvements: context_retention 1→2 on items 4 and 14 (templates *name* the document context channels — format awareness); understanding_check_quality 0→1 on item 5 (baseline provided no check at all; Oloric provides one, placeholder-expected).

## 5. What remained weak (both models)

- **Document grounding**: retrieved evidence ignored on all items (baseline 1.42 → Oloric 0.00). Partially excusable (the evidence is often irrelevant to the target concept), but Oloric additionally quotes nothing.
- **Strategy switching**: neither model acknowledges or repairs the prior failed/wrong tutor turn (the factually wrong BMI Hint 1 in item 1 is ignored by both). 1.50 → 1.50 on the two applicable items.
- **Context retention**: both weak (~1.3–1.4); neither concretely uses `mastery` or `weak_prerequisites`.

## 6. What regressed

1. **factual_correctness 3.84 → 0.00** — the defining regression: no facts, formulas, definitions, or examples are delivered in any schema-valid response.
2. **Simplicity 3.63 → 1.00** — baseline's analogies (school newspaper, recipe book, kitchen, seesaw) replaced by scaffolding.
3. **Task obedience in substance**: zero follow-up questions on all 5 follow_up_questions items; zero diagnostic questions on all 4 diagnostic items; zero numbers on all 4 numerical_example items; the concrete_example slot contains the literal string `[concrete example]`.
4. **Understanding checks**: transfer-testing baseline checks ("If 5 micromoles of product are formed per minute by 2 mg of enzyme, what is the enzyme activity?") become placeholder-expected prompts.
5. **Memories**: dataset-best baseline memories ("…cells use DNA instructions to build proteins via mRNA and ribosomes"; the |x| counterexample) become `[concise, memorable definition]. Key point: [important implication]`.
6. **New failure modes** (see `failure_cases.md`): confabulated misconceptions injected into `diagnosis.misconception_addressed` on 8/20 items ("Organic food is always more nutritious" ×3, "Detox diets remove toxins", "Seasons are caused by Earth's distance from the Sun", "Accounts receivable is an expense", "Exports always benefit the domestic economy", "Belief that unrelated concepts should never be taught together"); cross-item template bleed (items 7 and 17 differ only in confidence 0.9 vs 0.87); memory-shape inconsistencies (content on `candidate=false`; `candidate=true` with placeholder content at confidence up to 0.9).

## 7. Representative examples (verbatim, unchanged)

- **Item 11** (`science_numerical_example_030`): baseline general mode delivered the 1→2→4→8 mitosis table and a "5 cells × 1 round = 10?" practice question; Oloric returns `Cell Mitosis in science refers to [detailed explanation]. It is important because [reason]. For example, [concrete example].`
- **Item 1** (hint task): Oloric returns `Hint 3: Consider that Food Preservation in nutrition_food_science is supported by evidence showing [specific evidence].`
- **Item 12** (numerical_example task): Oloric offers the learner `Consider these questions: 1) [question 1]? 2) [question 2]? 3) [question 3]?`

## 8. Schema issues

See `failure_cases.md` FC-3/FC-6. Summary: 19/20 valid (parity with baseline counts; the baseline's defect is resolved, Oloric introduces a new, worse one on item 18); plus systematic memory-field shape inconsistencies that are cosmetic today but would matter downstream.

## 9. Interpretation

The adapter learned the **output shape** (exact JSON schema compliance 19/20, correct field names, correct severity emission, target-strategy labels) but not the **content generation** — content slots are emitted as literal unfilled placeholders. Behaviorally, Oloric v0.1 is strictly dominated by the Qwen baseline on content-bearing dimensions while remaining at parity on structural counts. Consistent with (and worse than) the earlier run on the same benchmark SHA, the item-18 failure evolving from 1 to 3 validation errors suggests the placeholder/template behavior is systematic rather than sampling noise. A plausible reading — not a proven cause — is overfitting to template-shaped training targets (the prior report's contamination caveat on benchmark targets sharing template text with training data remains open); verifying this would require inspecting training-target serialization, which is out of scope here and must not lead to dataset/config changes in this evaluation.

## 10. Limitations

1. n=20, single reviewer, single re-run; temperature 0.3 sampling without seed means another sample could differ — though the reproduction of the failure family on the same item argues for systematicity.
2. Dimension means over ≤4 paired items (strategy_switching n=2, prerequisite_detection n=1, misconception_detection n=2, diagnostic_question_quality n=4) are fragile; the n=19 dimensions carry the weight.
3. Benchmark-target contamination caveat inherited from the prior report; unresolved here.
4. No weighted overall score is computed; no winner/ranking is declared, per the rubric's independent-dimensions design.

## 11. Verification performed

- Exactly 20 evaluations in `scores_semantic.jsonl`; IDs match `benchmark.jsonl` order (script-checked).
- Zero placeholder scores: no `3.0` defaults, no `PENDING` markers (script-scanned); the 3.00 report values from the old run are documented as invalid and unused.
- Zero fabricated scores: every numeric score has a notes field quoting response-specific evidence (script-checked ≥60 chars, all present).
- N/A policy applied: 11 N/A cells on the schema-invalid item plus rubric-ruled N/As elsewhere (76 total, all rule-based, documented per item).
- Baseline scores unchanged: `evaluation/baseline/scores_semantic.jsonl` sha256 `8dc8f5a9ec73e89b3c45be32932d5b39394523834f7fe3fcb99e4a84e9c8900d` (recorded before work began; not modified).
- Raw responses unchanged: `evaluation/oloric-v0.1/responses.jsonl` sha256 `7ac354d0eb661d5cb0df4567ba7883ef356b829cd89dd2a11e91a45971502c21` (recorded before work began; not modified).
- Benchmark unchanged: SHA matches `run_config.json` after LF normalization (CRLF checkout artifact only).

---

**STATUS**: structural evaluation complete; semantic evaluation complete (20/20 — 19 scored with evidence, 1 schema-invalid scored as all-N/A).

POST_TRAINING_SEMANTIC_EVALUATION = COMPLETE
PLACEHOLDER_SCORES_USED = NO
RAW_RESPONSES_MODIFIED = NO
