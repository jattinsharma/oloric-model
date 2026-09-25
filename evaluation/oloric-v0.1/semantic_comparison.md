# Oloric v0.1 vs. Qwen Baseline — Semantic Dimension Comparison

**Aligned on**: identical benchmark IDs (verified programmatically: benchmark `evaluation/benchmark.jsonl` = 20 IDs; baseline structured-mode rows in `evaluation/baseline/responses.jsonl` = same 20 IDs, same order).
**Baseline scores**: taken as-is from `evaluation/baseline/scores_semantic.jsonl` (structured-mode rows only) — **not recomputed, not altered**.
**Oloric scores**: **PENDING** — raw Oloric responses were not persisted by the original run (only the schema-failure text survives in the report). Semantic scoring awaits the GPU re-run that exports `evaluation/oloric-v0.1/responses.jsonl`.

> **⚠️ Score-validity notice**: The automated `3.00` values in `evaluation_report_1790356740.json` are placeholder artifacts of `src/oloric/evaluation.py::_score_dimension()` (which returned a constant 3.0). They are **INVALID** and are not used anywhere in this comparison. The placeholder has since been removed; the automated harness now emits no semantic scores at all.

**Protocol**: no weighted overall rankings, no single "winner". Comparison is dimension-by-dimension. `N/A` marks dimensions the rubric declares not applicable for a given scenario (e.g. `strategy_switching` without prior failed tutor attempts, `memory_candidate_quality` when no memory candidate is present). `N/A` values are excluded from per-dimension averages.

---

## Legend

| Abbrev | Dimension |
|---|---|
| FC | factual_correctness |
| SIM | simplicity |
| TSS | teaching_strategy_selection |
| SWS | strategy_switching |
| MIS | misconception_detection |
| PRQ | prerequisite_detection |
| CTX | context_retention |
| DQ | diagnostic_question_quality |
| UC | understanding_check_quality |
| MC | memory_candidate_quality |
| DG | document_grounding |

---

## Aligned comparison table (baseline score / Oloric score)

Oloric column shows `—` until scored. `Δ` (Oloric − baseline) is intentionally left blank: a difference cannot be computed from a pending score.

| # | benchmark_id | FC | SIM | TSS | SWS | MIS | PRQ | CTX | DQ | UC | MC | DG | Oloric (11 dims) | Δ | Baseline evidence (condensed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | nutrition_food_science_hint_based_teaching_007 | 4 | 3 | 1 | 1 | N/A | N/A | 1 | N/A | 3 | 3 | 2 | PENDING | — | Task=hint; delivered full lecture; ignored wrong prior hint (BMI formula) |
| 2 | general_academic_concrete_example_040 | 4 | 4 | 4 | N/A | N/A | N/A | 2 | N/A | 4 | 3 | 2 | PENDING | — | Good bias example; transfer check; prereq 'basic literacy' unscaffolded |
| 3 | economics_diagnostic_questions_013 | 4 | 3 | 3 | N/A | N/A | 2 | 2 | 2 | 2 | N/A | 2 | PENDING | — | Q2 presupposes the flagged percentage-calculation gap; only 2 questions |
| 4 | nutrition_food_science_follow_up_questions_016 | 4 | 3 | 2 | N/A | N/A | N/A | 1 | N/A | 3 | 4 | 1 | PENDING | — | Re-taught concept instead of follow-ups; irrelevant evidence ignored |
| 5 | mathematics_follow_up_questions_035 | 3 | 4 | 1 | N/A | N/A | 1 | 1 | 1 | 0 | N/A | 1 | PENDING | — | 2-sentence non-diagnostic reply; no understanding check; mid-sentence self-correction |
| 6 | general_academic_simple_explanation_029 | 3 | 4 | 4 | N/A | N/A | N/A | 2 | N/A | 1 | 3 | 2 | PENDING | — | Conflated essay structure with logical argument structure; rote check |
| 7 | economics_numerical_example_029 | 4 | 4 | 2 | N/A | N/A | N/A | 1 | N/A | 3 | 3 | 1 | PENDING | — | Zero numbers for numerical_example; GDP/elasticity evidence ignored |
| 8 | nutrition_food_science_diagnostic_questions_009 | 4 | 4 | 3 | N/A | N/A | 1 | 1 | 2 | 2 | N/A | 1 | PENDING | — | 2 generic orientation questions; 'chemistry basics' prereq unprobed |
| 9 | nutrition_food_science_numerical_example_003 | 4 | 3 | 4 | N/A | N/A | N/A | 1 | N/A | 4 | 4 | 1 | PENDING | — | Correct µmol/min/mg units; genuine transfer check; irrelevant evidence |
| 10 | mathematics_misconception_detection_012 | 4 | 3 | 4 | N/A | 4 | N/A | 3 | N/A | 4 | 4 | 2 | PENDING | — | \|x\| counterexample; rigorous correction; near-ideal structured response |
| 11 | science_numerical_example_030 | 4 | 4 | 2 | N/A | N/A | N/A | 1 | N/A | 3 | 3 | 1 | PENDING | — | Described mitosis qualitatively; no 1→2→4→8 numbers (general mode had them) |
| 12 | nutrition_food_science_numerical_example_001 | 4 | 4 | 2 | N/A | N/A | N/A | 1 | N/A | 3 | 3 | 1 | PENDING | — | Salt osmosis correct but qualitative; memory slightly inconsistent with response |
| 13 | general_academic_misconception_detection_008 | 4 | 3 | 2 | 2 | 3 | N/A | 2 | N/A | 3 | 3 | 2 | PENDING | — | action=explain instead of misconception_correction; shallow 'why' |
| 14 | science_diagnostic_questions_013 | 4 | 4 | 3 | N/A | N/A | 1 | 1 | 2 | 2 | N/A | 1 | PENDING | — | 2 generic questions; no law-specific probes; expected_answer literal 'null' |
| 15 | science_follow_up_questions_020 | 4 | 4 | 2 | N/A | N/A | N/A | 1 | N/A | 3 | 4 | 1 | PENDING | — | action=explain; full re-teach; single recall check; excellent analogy memory |
| 16 | accountancy_diagnostic_questions_003 | 4 | 4 | 3 | N/A | N/A | 1 | 1 | 2 | 2 | N/A | 1 | PENDING | — | 2 generic questions; 'basic arithmetic' prereq unaddressed |
| 17 | economics_follow_up_questions_009 | 4 | 3 | 2 | N/A | N/A | 3 | 2 | N/A | 2 | N/A | 2 | PENDING | — | action=diagnose (prereq check) instead of follow-ups; prereq identification itself excellent |
| 18 | general_academic_practice_questions_008 | 4 | 4 | 2 | 2 | N/A | N/A | 2 | N/A | 3 | 3 | 2 | **N/A (schema-invalid)** | — | Baseline: explain-instead-of-practice. Oloric: schema-invalid (root-level `question`); all dims N/A, not scored |
| 19 | accountancy_prerequisite_detection_014 | 3 | 4 | 3 | N/A | N/A | 3 | 2 | N/A | 1 | N/A | 2 | PENDING | — | Correct prereq redirect; D2E 'ratio not percentage' imprecision; trivial 5+3 check |
| 20 | nutrition_food_science_follow_up_questions_035 | 4 | 4 | 2 | N/A | N/A | N/A | 1 | N/A | 3 | 3 | 1 | PENDING | — | action=explain; single check; over-broad anchor; confabulated 'You mentioned BMI earlier' |

---

## Per-dimension counts and baseline reference (baseline-only, no Oloric comparison yet)

Computed from the 20 baseline structured rows above. **These are baseline reference values only** — they will be paired with Oloric averages in the final report once real Oloric scores exist. No claim about improvement or regression is made from this table.

| Dimension | Baseline mean (applicable) | Baseline n | Score distribution |
|---|---|---|---|
| factual_correctness | 3.85 | 20 | 17×4, 3×3 |
| simplicity | 3.70 | 20 | 14×4, 6×3 |
| teaching_strategy_selection | 2.55 | 20 | 4×4, 4×3, 10×2, 2×1 |
| strategy_switching | 1.67 | 3 | 2×2, 1×1 |
| misconception_detection | 3.50 | 2 | 1×4, 1×3 |
| prerequisite_detection | 1.71 | 7 | 2×3, 1×2, 4×1 |
| context_retention | 1.45 | 20 | 2×3, 6×2, 12×1 |
| diagnostic_question_quality | 1.80 | 5 | 3×2, 1×1, 1×0 |
| understanding_check_quality | 2.55 | 20 | 3×4, 8×3, 7×2, 1×1, 1×0 |
| memory_candidate_quality | 3.31 | 13 | 4×4, 9×3 |
| document_grounding | 1.45 | 20 | 4×2, 16×1 |

---

## What the final comparison will report (after re-run)

Once `evaluation/oloric-v0.1/responses.jsonl` exists and the 19 valid responses are scored:

1. The **Oloric** and **Δ** columns will be filled with rubric scores and per-dimension differences.
2. Differences will be reported **dimension-by-dimension** (e.g. "TSS: baseline 2.55 → Oloric X"), never as a single weighted winner.
3. `#18` will remain **N/A (schema-invalid)** on the Oloric side for all dimensions, with a structural note instead of a semantic score — consistent with the no-silent-repair protocol.
4. Rows where Oloric's response structure legitimately changes applicability (e.g. Oloric emits a memory candidate where the baseline had none, or vice versa) will be re-evaluated for N/A status against the rubric **per Oloric response**, not copied from the baseline.

STATUS: baseline side complete and evidence-linked; Oloric side PENDING_RE_RUN.
