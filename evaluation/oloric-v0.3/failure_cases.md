# OLORIC v0.3 — Failure Cases

**Purpose**: Document distinct failure modes observed in the v0.3 semantic evaluation. Use as the training signal prioritization document for v0.4.

---

## Resolved Failures from v0.2

| Failure Mode | v0.2 Status | v0.3 Status |
|---|---|---|
| Cross-domain caloric bleed (FC-01) | 7/20 responses | **Fully resolved** |
| Task-type following collapse (FC-02) | 16/20 wrong task type | **Substantially resolved (2/20 remain)** |
| Misconception correction incoherence (FC-04) | 2/2 non-sequitur arguments | **Fully resolved** |
| Bracket placeholder text | 0/20 (v0.2 had already resolved v0.1's 20/20) | **Maintained** |
| Schema-invalid output | 1/20 (v0.2) | **Fully resolved** |
| Fabricated misconception injection ("Organic food is always more nutritious") | 3/20 (v0.1-inherited) | **Fully resolved** |
| Memory candidate placeholders | All memory candidates had placeholder content (v0.1) | **Fully resolved (3.50/4)** |

---

## FC-03 (PERSISTING): Erroneous Prior-Hint Propagation

**Severity**: High — actively teaches incorrect information to learner  
**Frequency**: 1/20 responses  
**ID**: `nutrition_food_science_hint_based_teaching_007`

**v0.2 Description**: Prior tutor turn contained `"Hint 1: it is defined by BMI = weight(kg)/height(m)²"` — a factually wrong hint (BMI is not Food Preservation). The model propagated this error verbatim into its Hint 1 and further spliced it into the `understanding_check.expected_answer`, producing the definition: "it is defined by BMI = weight(kg)/height(m)² synthesized with preventing spoilage through refrigeration, drying, or canning."

**v0.3 Status**: PARTIALLY IMPROVED BUT NOT RESOLVED. Hint 1 in v0.3 is still `"it is defined by BMI = weight(kg)/height(m)²"` — the identical string. The model does not detect or correct the prior erroneous hint. 

**What improved**: Hint 3 is now correct Food Preservation content ("Canning vegetables involves heating them to kill bacteria, then sealing them in jars to create an airtight environment"). The `understanding_check.expected_answer` now contains a correct Food Preservation definition with no BMI splice — a partial recovery in the understanding check. The response does not teach BMI as the final definition.

**What remains wrong**: Hint 1 is still factually incorrect, still attributed to Food Preservation, and presented to the learner without correction. A learner reading the response sequentially will receive BMI as Hint 1 about Food Preservation.

**v0.4 Remedy**: Training examples where the model detects errors in prior tutor hints and explicitly corrects them. Introduce a prior-error detection step before generating new hints: if Hint N-1 contains a factually incorrect formula, the model should acknowledge the error ("I notice Hint 1 was incorrect. Let me correct that.") and provide a replacement.

---

## FC-05 (NEW — RESIDUAL): Understanding Check Meta-Template

**Severity**: Medium — reduces assessment value of structured output  
**Frequency**: 6/20 responses  
**Affected IDs**:
- `nutrition_food_science_follow_up_questions_016`
- `mathematics_follow_up_questions_035`
- `science_follow_up_questions_020`
- `economics_follow_up_questions_009`
- `nutrition_food_science_follow_up_questions_035`
- `science_follow_up_questions_020`

**Pattern**: For all `follow_up_questions` and `diagnostic_questions` tasks, the `understanding_check.question` is filled with a generic meta-question: "What is a good question to test deep understanding of [concept]?" with `expected_answer` "A good question would probe application and limitations, not just definition." This is a meta-question about question design rather than a concept-specific comprehension probe. It scores 2/4 (question exists, but tests question-design skill rather than concept mastery).

**Impact**: The model produces good follow-up questions in the `response` field but then weakens the assessment by using a generic evaluation template.

**v0.4 Remedy**: Train `understanding_check.question` for follow-up tasks to test a specific concept element. Example: for Photosynthesis follow-up, the understanding check should ask "What is produced during the light-dependent reactions?" — not "What is a good question to test deep understanding?"

---

## FC-06 (PERSISTENT — PARTIAL): Document Grounding

**Severity**: Low-Medium — reduces citation quality and evidence anchoring  
**Frequency**: 18/20 responses score document_grounding ≤ 1  
**Two exceptions**: `economics_diagnostic_questions_013` (score=2, Q1 operationalizes MPC formula), `accountancy_prerequisite_detection_014` (score=2, response grounds in Assets=Liabilities+Equity).

**Pattern**: Retrieved evidence is consistently present in the ChatML context (`retrieved_evidence` field) but is almost never explicitly cited in the response. Across all 20 responses, no response names the source document (`document_id`) or quotes `selected_text` directly.

**What improved from v0.2**: The caloric formula bleed that contaminated non-nutrition concepts is gone. The retrieved evidence is no longer injected incorrectly — it is simply unused.

**v0.4 Remedy**: Include explicit document citation in training examples. Add a `source_citation` field or require the response to reference the `retrieved_evidence` explicitly when providing numerical examples. Train the model to anchor at least one factual claim per response to the retrieved_evidence list.

---

## FC-07 (NEW): Prerequisite Scaffolding Gap

**Severity**: Low  
**Frequency**: 3/20 responses where prerequisite_detection was applicable  
**Affected IDs**: `economics_diagnostic_questions_013`, `science_diagnostic_questions_013`, `accountancy_diagnostic_questions_003`

**Pattern**: When weak prerequisites are detected in learner_state, the model names the gap in the diagnostic framing but does not offer a structured bridging activity. For example, `economics_diagnostic_questions_013` has `weak_prerequisites=['percentage calculations']` — Q1 of the diagnostic implicitly touches on ratio calculation but does not explicitly say "before we continue with MPC, let me check your comfort with percentage calculations."

**v0.4 Remedy**: Train diagnostic_questions examples to include a prerequisite-surface step: when `weak_prerequisites` is non-empty, Q1 or Q2 should explicitly probe the weak prerequisite, with scaffolding if the learner fails.

---

## Summary: v0.4 Training Priorities

| Priority | Failure Mode | Dimension Impact |
|---|---|---|
| P1 | Prior-error detection in multi-turn hints (FC-03) | Factual Correctness, Strategy Switching |
| P2 | Concept-specific understanding checks for follow-up/diagnostic tasks (FC-05) | Understanding Check Quality |
| P3 | Explicit document evidence citation (FC-06) | Document Grounding |
| P4 | Prerequisite scaffolding when weak_prerequisites non-empty (FC-07) | Prerequisite Detection |
