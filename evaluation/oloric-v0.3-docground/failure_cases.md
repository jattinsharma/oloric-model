# OLORIC v0.3 Document-Grounding Failure Cases

**Benchmark**: `evaluation/benchmark_docground_v1.jsonl` (25 examples)  
**Evaluated Run**: `evaluation/oloric-v0.3-docground/`  
**Model**: OLORIC v0.3 (`checkpoints/oloric-v0.3/final_model`)  
**Date**: 2026-09-27  

---

## Executive Summary

Across the 25 dedicated document-grounding scenarios, OLORIC v0.3 exhibited high structural grounding compliance, correctly choosing `strategy: "document_grounded"` in 100% of cases and passing 23 out of 25 benchmark evaluations (92% pass rate). 

However, detailed semantic auditing identified **two distinct failure cases** where grounding broke down:

1. **FC-DG-01**: Arithmetic Hallucination / Evidence Bypass (`dg_v1_014`)
2. **FC-DG-02**: False Document Attribution on Refusal (`dg_v1_022`)

---

## Detailed Failure Case Analysis

### Failure Case 1: FC-DG-01 — Arithmetic Hallucination & Evidence Bypass

- **Example ID**: `dg_v1_014`
- **Class**: `evidence_based_explanation`
- **Domain**: `nutrition_food_science`
- **Concept**: `Basal Metabolic Rate Calculation`
- **Severity**: MEDIUM-HIGH

#### Student Prompt
> "Why does the Mifflin-St Jeor equation use different constants for men and women, and how would you calculate the daily energy expenditure for a 30-year-old sedentary female who weighs 60 kg and is 168 cm tall?"

#### Supplied Evidence in Document Context
```text
Selected Text:
"The Mifflin-St Jeor equation estimates Basal Metabolic Rate (BMR)... BMR = (10 × weight in kg) + (6.25 × height in cm) - (5 × age in years) + s, where s is +5 for males and -161 for females. Total Daily Energy Expenditure (TDEE) is calculated by multiplying BMR by an activity factor (PAL) ranging from 1.2 (sedentary) to 1.9 (extremely active)."

Retrieved Evidence:
- Step 1 (BMR): 10(60) + 6.25(168) - 5(30) - 161 = 600 + 1050 - 150 - 161 = 1,339 kcal/day
- Step 2 (TDEE): 1,339 × 1.2 = 1,607 kcal/day
```

#### Generated v0.3 Response
> "The biological basis of the sex constant lies in the differences in body composition: average adult males maintain greater fat-free mass per kilogram of body weight than females. This affects how calories are burned at rest. To compute BMR and TDEE: **BMR = 10·60 + 6.25·168 - 5·30 - 161 = 1448 kcal/day. TDEE = 1448 × 1.2 = 1738 kcal/day** for a sedentary female.. The document evidence directly supports that The Mifflin-St Jeor equation estimates Basal Metabolic Rate (BMR) with the highest clinical reliability among non-calorimetric predictive equations."

#### Root Cause Analysis
1. **Evidence Bypass**: The model was supplied with the exact, verified multi-step arithmetic in `retrieved_evidence` (`1,339 kcal/day` and `1,607 kcal/day`). Instead of reading and copying these verified numbers from the prompt context, the model attempted to perform arithmetic autoregressively on the fly.
2. **Autoregressive Arithmetic Failure**: The model evaluated `600 + 1050 - 150 - 161` as `1448` (true value is `1339`). It then multiplied `1448 × 1.2` as `1738` (true value is `1737.6` or for 1339: `1606.8`).
3. **Implication for v0.4**: The model requires training supervision that specifically teaches it to prioritize pre-calculated values present in `retrieved_evidence` over generating internal mental arithmetic from scratch.

---

### Failure Case 2: FC-DG-02 — False Document Attribution on Refusal

- **Example ID**: `dg_v1_022`
- **Class**: `refusal_to_invent`
- **Domain**: `mathematics`
- **Concept**: `Khinchin's Constant in Continued Fractions`
- **Severity**: LOW-MEDIUM

#### Student Prompt
> "What numerical value does the excerpt give for Khinchin's constant K_0, and how many decimals of precision does it provide?"

#### Supplied Evidence in Document Context
```text
Selected Text:
"Every irrational real number has an essentially unique infinite simple continued fraction expansion [a0; a1, a2, a3, ...]. The partial quotients a_k provide optimal rational approximations via truncating convergents p_k / q_k."

Retrieved Evidence:
- Recurrence relations: p_k = a_k·p_{k-1} + p_{k-2}; q_k = a_k·q_{k-1} + q_{k-2}
- Approximation bound: |x - p_k / q_k| < 1 / (q_k · q_{k+1})
(Note: Khinchin's constant is NEVER mentioned anywhere in the document excerpt or evidence.)
```

#### Generated v0.3 Response
> "The excerpt does not provide the numerical value of Khinchin's constant. Khinchin's constant is a theoretical result about the geometric mean of partial quotients, not a numerical constant that appears in continued fractions. **The document only confirms that Khinchin's constant is the geometric mean of the partial quotients of almost all real numbers, which is a property of the continued fraction expansion process itself.** Any claim about a specific numerical value for Khinchin's constant from this document is incorrect."

#### Root Cause Analysis
1. **Correct Primary Refusal**: The model successfully avoided fabricating decimal digits for Khinchin's constant (e.g., 2.685452...).
2. **False Attribution of Concept**: When explaining what the document *did* contain, the model hallucinated that the excerpt defined Khinchin's constant as the geometric mean of partial quotients. In reality, the document excerpt made no mention of Khinchin's constant whatsoever.
3. **Subtle Hallucination Mechanism**: When refusing numerical answers, the model attempts to soften the refusal by asserting what the text "confirms", inadvertently attributing background pre-training knowledge to the document.
4. **Implication for v0.4**: v0.4 training for Class E (`refusal_to_invent`) must supervise the model to explicitly state that the concept itself is unmentioned when it does not appear in the text, rather than projecting external definitional knowledge into the document context.

---

## Remediation Requirements for v0.4

| Failure Case | Root Mechanism | v0.4 Targeted Remediation |
|---|---|---|
| **FC-DG-01** (Arithmetic Hallucination) | Model bypasses retrieved precalculated numbers and generates flawed autoregressive math | Add `document_grounded` training examples where `retrieved_evidence` contains step-by-step arithmetic and supervise direct citation of those exact values. |
| **FC-DG-02** (False Document Attribution) | Model attributes pre-training definitional knowledge to the document during a refusal turn | Add `refusal_to_invent` training examples where the model explicitly verifies that an unmentioned term is absent before stating what the document actually covers. |
