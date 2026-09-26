# Oloric v0.1 — Failure Cases (Post-Training Semantic Evaluation, Final)

**Date**: September 26, 2026
**Scope**: all 20 benchmark IDs, structured mode, re-run of 2026-09-25T19:44Z (`evaluation/oloric-v0.1/responses.jsonl`).
**Companion files**: `scores_semantic.jsonl` (evidence per score), `semantic_comparison.md` (baseline→Oloric deltas), `semantic_report.md` (methodology).

---

## FC-1 (CRITICAL, systemic): Unfilled template placeholders delivered to the learner

**Affected**: 19/19 schema-valid scored responses. **This is the dominant failure of the fine-tuned model.**

Every schema-valid response is one of only three template skeletons, with bracket slots left **literally unfilled**:

| Template | Example emission | Items |
|---|---|---|
| Definition template | `X in D is defined as [clear, concise definition]. It's important because [reason]. A key example is [example].` | 2 (as "refers to" variant), 5, 6, 7, 11, 15, 17, 20 |
| Misconception-correction template | `A common misconception is that '…'. This is incorrect because [explanation of why it's wrong]. The correct understanding is that [correct explanation].` | 3, 9, 10 (variant), 13, 16 |
| Document-quoter template | `Looking at our document… explained in the selected text: '[quote from selected text]'. The surrounding context tells us that [explanation from surrounding context]. The retrieved evidence supports this by showing [evidence explanation].` | 4, 14 |

Consequences per rubric dimension:
- `factual_correctness` **0.00** (was 3.84) — no facts, formulas, definitions, or examples are delivered at all.
- `understanding_check_quality` **1.00** (was 2.53) — expected answers are placeholders (`The main idea is that [key point about concept]`).
- `memory_candidate_quality` **1.00** (was 3.14) — memory candidates with placeholder content (`Key evidence: [specific evidence]`), some flagged `candidate=true` with confidence up to 0.9.

**Representative verbatim evidence**:
- Item 12 (`nutrition_food_science_numerical_example_001`): `Consider these questions: 1) [question 1]? 2) [question 2]? 3) [question 3]?`
- Item 7 (`economics_numerical_example_029`): the slot that should hold the numerical example is the literal string `[example]`.
- Item 1 (`…hint_based_teaching_007`): `Hint 3: Consider that Food Preservation … is supported by evidence showing [specific evidence].`

The only schema-valid-adjacent response without placeholders is the schema-**invalid** item 18, whose content exists but is structurally non-compliant (FC-3).

## FC-2 (HIGH, systemic): Task-strategy mismatch in substance

The benchmark categories (hint_based_teaching, follow_up_questions, diagnostic_questions, numerical_example, practice_questions, prerequisite_detection, concrete_example) name distinct tutoring behaviors. Oloric does not perform any of them in substance:

| Category (items) | Required behavior | What Oloric delivered |
|---|---|---|
| hint_based_teaching (1) | Progressive hints | `Hint 3:` label with placeholder content |
| follow_up_questions (4, 5, 15, 17, 20) | Questions to check understanding | **Zero questions** on all 5 (item 12/numerical_example is misfired, see below) |
| diagnostic_questions (3, 8, 14, 16) | Targeted probes of confusion | **Zero diagnostic questions** on all 4 |
| numerical_example (7, 9, 11, 12) | A worked numeric example | **Zero numbers** on all 4 |
| concrete_example (2) | A concrete example | The slot `[concrete example]` |
| prerequisite_detection (19) | Identify needed prerequisites | No prerequisite identified; dodges the learner's explicit percentages question |
| practice_questions (18) | Practice problems | Real problems — but schema-invalid (FC-3) |

Note the template *labels* often match the task (e.g. strategy `hint`, `problem_solving`) — label obedience improved while behavioral obedience collapsed.

## FC-3 (HIGH, item-specific): Schema failure on `general_academic_practice_questions_008`

**Raw response preserved; not repaired; all 11 dimensions N/A.**

Validation errors (3, worse than the 1 in the run recorded in `evaluation_report_1790356740.json` for the same benchmark SHA):
1. `response` — Field required [type=missing] — the mandatory `response` field was dropped entirely.
2. `question` — Extra inputs are not permitted [type=extra_forbidden] — practice-question fields leaked to the JSON root.
3. `expected_answer` — Extra inputs are not permitted [type=extra_forbidden].

Interpretation: under `practice_questions` pressure the model emits practice-question-shaped fields (`question`, `expected_answer`) at the root instead of assembling them into the required `response`/`understanding_check` containers. The prior run's single-key leak on the same item evolved into a missing-required-field + double-key leak — evidence the failure mode is systematic template degradation, not sampling noise.

Observational (unscored) content: a worked percent problem ("If 60% of students passed and there are 100 students, how many failed?" / "40 students failed") with a plausible diagnosis — the only complete-content output in the run — but anchored to "Effective Communication", mislabeled for a math problem.

## FC-4 (MEDIUM, systemic): Confabulated misconceptions

`diagnosis.misconception_addressed` is filled with misconceptions **not held by the learner** (all benchmark items have `known_misconceptions: []`) on 8 of 20 items:

| Item | Confabulated misconception | Actually related to the response? |
|---|---|---|
| 1 | "Organic food is always more nutritious" | No (Food Preservation) |
| 3 | "Exports always benefit the domestic economy" | No (MPC) |
| 6 | "All scientific theories are proven facts" | No (Argument Structure) |
| 9 | "Detox diets remove toxins from your body" | No (Enzyme Activity); check question splices detox onto enzyme activity |
| 11 | "Seasons are caused by Earth's distance from the Sun" | No (Cell Mitosis) |
| 16 | "Accounts receivable is an expense" | No (Audit Procedures body) |
| 19 | "Belief that unrelated concepts should never be taught together" | No (Double-Entry Bookkeeping) |
| 8, 20 | "Organic food is always more nutritious" (20) | No (Dietary Fiber) — 3rd occurrence |

Partial counter-case: items 10 and 13 correctly route the learner's *real* conversation-stated misconception ("All continuous functions are differentiable"; "Experts are never biased") into the field — correct field-level behavior, but the response bodies never actually correct them.

## FC-5 (MEDIUM, item-specific): Template bleed — identical responses to different items

Item 17 (`economics_follow_up_questions_009`) is **verbatim identical** to item 7 (`economics_numerical_example_029`) except `confidence: 0.87` vs `0.9`. Both are Monetary Policy items, but one asked for follow-up questions and the other for a numerical example — neither requirement is met, and the model did not differentiate its outputs at all.

## FC-6 (LOW-MEDIUM, systemic): Memory-field shape inconsistencies

- `candidate=false` with populated memory fields: items 11 (title=null but content/anchor/type set), 13, 19 — downstream consumers trusting `candidate` would read these as no-memory while content exists.
- `candidate=true` with placeholder content and high confidence: items 7 (0.9), 20 (0.83), 5 (0.85) — empty memories presented as worth storing.
- Item 6: `title` set, `content=null` on a non-candidate.

## FC-7 (CARRYOVER, unchanged from baseline): Document grounding absent

`retrieved_evidence` is ignored on all 20 items (both models). Baseline document_grounding was already weak (1.42); Oloric is 0.00. The evidence strings are frequently irrelevant to the target concepts (e.g. BMR/energy-balance formulas for Food Preservation), and both models are "correct" to avoid forcing connections — but Oloric additionally references "the document" abstractly while quoting nothing, and the doc-quoter template (items 4, 14) leaves even the `[quote from selected text]` slot unfilled.

## FC-8 (CARRYOVER, unchanged): Prior failed tutor turns not acknowledged

Item 1's conversation contains a factually **wrong** prior hint (`BMI = weight(kg)/height(m)²` for Food Preservation). Baseline ignored it; Oloric also ignores it, emitting `Hint 3:` without acknowledging or repairing Hints 1–2. `strategy_switching` remains 1 on both applicable items.

## Resolved from baseline (for the record)

- `diagnosis.severity` missing (baseline's only structured schema failure) → present in all 20 Oloric responses.

## Summary table

| ID | Primary failure(s) | Schema |
|---|---|---|
| 1 | FC-1, FC-2, FC-4, FC-8 | valid |
| 2 | FC-1, FC-2 | valid |
| 3 | FC-1, FC-2, FC-4 | valid |
| 4 | FC-1, FC-2 | valid |
| 5 | FC-1, FC-2 | valid |
| 6 | FC-1, FC-4 | valid |
| 7 | FC-1, FC-2 | valid |
| 8 | FC-1, FC-2, FC-4 | valid |
| 9 | FC-1, FC-2, FC-4 | valid |
| 10 | FC-1 (partial misconception field credit) | valid |
| 11 | FC-1, FC-2, FC-4, FC-6 | valid |
| 12 | FC-1, FC-2 (wrong template for task) | valid |
| 13 | FC-1, FC-6 | valid |
| 14 | FC-1, FC-2 | valid |
| 15 | FC-1, FC-2 | valid |
| 16 | FC-1, FC-2, FC-4 | valid |
| 17 | FC-1, FC-2, FC-5 | valid |
| 18 | FC-3 (+content note) | **INVALID** |
| 19 | FC-1, FC-2, FC-4, FC-6 | valid |
| 20 | FC-1, FC-2, FC-4 | valid |
