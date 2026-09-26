# Oloric v0.2 — Failure Cases

**Purpose**: Document distinct failure modes observed in the v0.2 structured evaluation. Use this as the training signal prioritization document for v0.3.

---

## FC-01: Cross-Domain Caloric Example Bleed *(Critical Systemic)*

**Severity**: High — produces factually false scientific claims  
**Frequency**: 7 of 20 responses  
**Affected IDs**:
- `nutrition_food_science_numerical_example_003` (Enzyme Activity ← caloric diet formulas)
- `nutrition_food_science_numerical_example_001` (Food Preservation ← caloric diet formulas)
- `science_numerical_example_030` (Cell Mitosis ← chemistry stoichiometry + physics impulse)
- `science_diagnostic_questions_013` (Newton's Laws ← chemistry stoichiometry)
- `nutrition_food_science_diagnostic_questions_009` (Nutrient Deficiencies ← caloric accounting)
- `mathematics_follow_up_questions_035` (Matrix Operations ← quadratic + calculus examples)
- `nutrition_food_science_follow_up_questions_035` (Dietary Fiber ← caloric diet formulas)

**Pattern**: The model has learned a highly reusable `concrete_example` template anchored to nutrition/caloric formulas (`2000-cal diet, 50% carbs = 1000 cal` and `500-cal deficit = 1lb/week`) and applies it across unrelated domains, attributing these formulas to the wrong concept in the response body.

**Root Cause**: Training dataset likely overrepresents nutrition-domain examples as the default `concrete_example` template. The model treats the concept name as a slot but reuses the fixed example body.

**Distinguishing mark from v0.1**: v0.1 emitted these as literal `[concrete example]` placeholders. v0.2 fills them with real numbers — making the error more credible and therefore more harmful (a learner might accept `2000-calorie diet` as an Enzyme Activity illustration).

**v0.3 Remedy**: Per-concept concrete example training data. No cross-domain caloric formulas in non-nutrition examples. Diverse domain-specific numerical examples across all 5 domains.

---

## FC-02: Task-Type Ignoring — Systematic concrete_example Override *(Critical Systemic)*

**Severity**: High — task instructions are not followed  
**Frequency**: 13 of 20 responses select `concrete_example`/`explain`; 12+ of these do not match the requested task
**Affected task types**:
- `follow_up_questions` (4 examples) → all produce 0 follow-up questions; all use concrete_example
- `diagnostic_questions` (4 examples) → 3 produce 0 diagnostic questions; only `science_follow_up_questions_020` succeeds
- `numerical_example` (6 examples) → 3 use domain-wrong examples
- `practice_questions` (1 example) → partially correct
- `hint_based_teaching` (1 example) → partially correct

**Pattern**: The training distribution strongly favors the `action=explain, strategy=concrete_example` pattern regardless of the task type field. The model has memorized the structural schema but not the task-conditional output variation.

**v0.3 Remedy**: Training examples must balance all task types. Equal representation of `diagnostic_questions`, `follow_up_questions`, `practice_questions`, `hint_based_teaching`, and `numerical_example` tasks in training data. Consider task-type conditioning in the prompt template.

---

## FC-03: Erroneous Prior-Hint Propagation *(Critical, 1 example)*

**Severity**: High — actively teaches incorrect information to learner  
**ID**: `nutrition_food_science_hint_based_teaching_007`  
**Detail**: The conversation context contains a prior tutor hint: "Hint 1: it is defined by BMI = weight(kg)/height(m)²" — which is the BMI formula, NOT a Food Preservation formula. This was the same error documented in v0.1. v0.2 perpetuates this error: Hint 1 in the new response is identical: "it is defined by BMI = weight(kg)/height(m)²". The expected_answer in the understanding_check then splices this wrong formula into the Food Preservation definition: "it is defined by BMI = weight(kg)/height(m)² synthesized with preventing spoilage through refrigeration, drying, or canning."

**Impact**: A learner who reads this response will be told that BMI = Food Preservation. The model should have detected that Hint 1 was factually wrong and either corrected it or produced a replacement hint sequence.

**v0.3 Remedy**: Include training examples where the model must detect and correct prior erroneous hints. Introduce explicit prior-error-detection behavior in hint-based teaching examples.

---

## FC-04: Incoherent Misconception Correction Argument *(Major, 2 examples)*

**Severity**: Medium — misconception correctly identified but correction is logically invalid  
**IDs**:
- `mathematics_misconception_detection_012`: misconception = "All continuous functions are differentiable" → correction argument invokes Linear Algebra as the disproof, which is not logically connected
- `general_academic_misconception_detection_008`: misconception = "Experts are never biased" → correction invokes spaced repetition and active recall, which is about learning strategies, not expert bias

**Pattern**: The model correctly identifies the target misconception (improvement from v0.1), but the correction body is constructed by filling in the adjacent concept definition from the document (e.g., Linear Algebra, Learning Strategies) rather than producing a domain-appropriate counterargument.

**v0.3 Remedy**: Misconception correction training examples that include valid counterarguments (e.g., |x| as counterexample for differentiability, cognitive bias research for expert bias). Do not use adjacent document concept as the correction argument.

---

## FC-05: Fabricated Misconception Injection via diagnosis Field *(Persistent)*

**Severity**: Low (field-level, not in response body) — misleading metadata  
**Frequency**: 11 of 20 responses where the task is NOT misconception-correction still populate `diagnosis.misconception_addressed` with generic metacognitive claims:
- `"Belief that abstract concepts cannot be understood without examples"` (5 occurrences)
- `"Belief that numerical examples aren't helpful for understanding abstract concepts"` (5 occurrences)
- `"Organic food is always more nutritious"` (0 occurrences in v0.2 — eliminated from v0.1's 3 occurrences)

**Pattern**: The model fills `misconception_addressed` with a template string regardless of whether a real misconception was stated. The `"Organic food is always more nutritious"` confabulation from v0.1 is eliminated, replaced by generic metacognitive injections that are at least not factually harmful.

**v0.3 Remedy**: Train `misconception_addressed = null` when `known_misconceptions = []` in learner state and the task is not misconception-correction.

---

## FC-06: Memory Candidate Content-Anchor Mismatch *(Moderate, 2 examples)*

**Severity**: Medium — stored memory will mislead future recall  
**IDs**:
- `nutrition_food_science_numerical_example_001`: memory anchor = "Food Preservation", content = caloric diet formula → stores a wrong example under the wrong concept
- `economics_follow_up_questions_009`: memory anchor = "Monetary Policy", content = MPC/fiscal multiplier example → stores fiscal policy examples labeled as Monetary Policy

**Pattern**: Memory candidates are now real (improvement from v0.1 placeholders), but the anchor-to-content mapping inherits the cross-domain bleed from FC-01.

**v0.3 Remedy**: Fix FC-01 (correct concept-specific examples) and memory candidate quality improves automatically.

---

## Resolved Failures from v0.1

| Failure Mode | v0.1 Status | v0.2 Status |
|---|---|---|
| Bracket placeholder text in response body | All 19 valid responses | **Fully resolved** |
| Schema-invalid response (`general_academic_practice_questions_008`) | 1 failure | **Fully resolved** |
| `diagnosis.severity` field absent | 1 case (original trigger) | **Fully resolved** |
| `"Organic food is always more nutritious"` confabulation injection | 3 occurrences | **Fully resolved** |
| Zero factual content (score=0) across the run | 19/19 valid responses | **Fully resolved** |
| All memory candidates = placeholders | 5/5 candidates | **Fully resolved** |

---

## Summary: v0.3 Training Priorities

| Priority | Failure Mode | Dimension Impact |
|----------|---|---|
| P1 | Per-concept domain-specific examples (fix FC-01) | Factual Correctness, Document Grounding |
| P2 | Task-type conditional generation (fix FC-02) | Teaching Strategy Selection |
| P3 | Prior-error detection in multi-turn hints (fix FC-03) | Context Retention, Factual Correctness |
| P4 | Misconception correction with valid counterarguments (fix FC-04) | Misconception Detection |
| P5 | Null misconception_addressed when no misconception present (fix FC-05) | Schema semantic correctness |
| P6 | Document grounding via retrieved_evidence citation | Document Grounding |
