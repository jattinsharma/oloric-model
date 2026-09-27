# OLORIC v0.3 Supplemental Document-Grounding Evaluation Report

**Evaluation Date**: 2026-09-27  
**Model**: OLORIC v0.3 (`checkpoints/oloric-v0.3/final_model`)  
**Base Model**: Qwen/Qwen3-4B-Instruct-2507 (4-bit QLoRA)  
**Benchmark**: `evaluation/benchmark_docground_v1.jsonl` (25 examples, 5 classes)  
**Artifacts Evaluated**:
- `evaluation/oloric-v0.3-docground/responses.jsonl` (25 responses)
- `evaluation/oloric-v0.3-docground/evaluation_report_1790485123.json`
- `evaluation/oloric-v0.3-docground/run_config.json`

---

## 1. Executive Summary

This report establishes the first empirical baseline of OLORIC v0.3 on the purpose-built 25-example document-grounding benchmark (`evaluation/benchmark_docground_v1.jsonl`).

In the original frozen 20-item benchmark (`evaluation/benchmark.jsonl`), v0.3 received a document grounding score of 1.15/4. As diagnosed in `docs/v0.4-root-cause-audit.md`, that low score was primarily a benchmark artifact: 0/20 items requested document grounding, and all document excerpts were generic placeholders (`"The concept of X is fundamental..."`).

On this dedicated document-grounding benchmark with rich academic excerpts and explicit grounding instructions, OLORIC v0.3 demonstrates **strong latent grounding capabilities**:
- **Strategy Selection**: 25/25 (100%) chose `strategy: "document_grounded"`
- **Schema Validity**: 25/25 (100%) valid JSON matching `OloricModelOutput`
- **Benchmark Pass Rate**: 23/25 (92.0%) passed the rubric
- **Aggregate Behavioral Score**: **3.75 / 4.00**
- **Failures Identified**: **2** (`FC-DG-01` in Class C, `FC-DG-02` in Class E)

---

## 2. Dimension Scores (0–4 Scale)

| Evaluation Dimension | Mean Score | Key Observations |
|---|:---:|---|
| **1. Evidence Use** | **3.80** | Extensive direct quotation of selected text and retrieval fields. 1 bypass in `dg_v1_014`. |
| **2. Faithfulness to selected_text** | **3.84** | High verbatim accuracy; quotes text faithfully without distorting author claims. |
| **3. Faithfulness to retrieved_evidence** | **3.76** | Robust integration of retrieved formulas and thresholds; 1 arithmetic bypass (`dg_v1_014`). |
| **4. Unsupported-Claim Avoidance** | **3.84** | High discipline against external hallucinations; 1 false attribution (`dg_v1_022`). |
| **5. Conflict Handling** (Class D) | **4.00** | 100% detection rate of evidence discrepancies across all 5 test scenarios. |
| **6. Grounded Explanation Quality** | **3.52** | Strong conceptual synthesis; occasional minor syntactic repetition. |
| **7. Learner Usefulness** | **3.76** | Clear, actionable responses directly addressing student questions. |
| **Overall Behavioral Grounding** | **3.75** | **Strong baseline (3.52 / 4.00)** |

---

## 3. Results by Behavior Class

| Behavior Class | N | Pass Rate | Mean Score | Strengths & Failure Modes |
|---|:---:|:---:|:---:|---|
| **A. `direct_evidence_use`** | 5 | 5/5 (100%) | **3.82** | Flawless extraction of formulas (MPC, Integration by Parts, Stoichiometry, Depreciation). |
| **B. `grounded_paraphrase`** | 5 | 5/5 (100%) | **3.76** | Excellent pedagogical simplification (e.g. 5-step revenue checklist, elasticity plain English). |
| **C. `evidence_based_explanation`** | 5 | 4/5 (80%) | **3.62** | Step-by-step arithmetic operationalization. **1 Failure**: `FC-DG-01` arithmetic hallucination on BMR (`dg_v1_014`). |
| **D. `conflict_handling`** | 5 | 5/5 (100%) | **3.84** | **Standout competence**: Model consistently detected and surfaced conflicts rather than silently picking one. |
| **E. `refusal_to_invent`** | 5 | 4/5 (80%) | **3.70** | Successfully refused 1982 interest rates, Pa-231 half-life, spirulina B12, and dropout statistics. **1 Failure**: `FC-DG-02` false attribution on Khinchin's constant (`dg_v1_022`). |

---

## 4. Deep Dive by Behavior Class

### Class A: Direct Evidence Use (Score: 3.82 / 4.00 | 5/5 Pass)
The model consistently identified the exact passage from `selected_text`, quoted it with formal citations (title, section, page), and extracted formulas accurately:
- `dg_v1_001`: Extracted theoretical multiplier 5.0 and realized range 1.2–1.8 from text.
- `dg_v1_002`: Extracted formula `∫ u dv = u·v - ∫ v du` and the complete LIATE acronym.
- `dg_v1_003`: Accurately quoted updated modern net yield of 30–32 ATP per glucose and 15 ATP from 6 NADH.
- `dg_v1_004`: Accurately quoted heme (15–35%) vs non-heme (2–20%) absorption and Fe3+ -> Fe2+ reduction.
- `dg_v1_005`: Accurately cited straight-line and DDB formulas with 200% rate.

### Class B: Grounded Paraphrase (Score: 3.76 / 4.00 | 5/5 Pass)
The model demonstrated strong ability to translate dense academic text into pedagogical language without losing meaning:
- `dg_v1_006`: Mapped all 4 price elasticity determinants into everyday terms (substitutes, essential vs extravagant, income bite, adjustment time).
- `dg_v1_008`: Simplified CRISPR-Cas9 mechanism into guide RNA and PAM restriction.
- `dg_v1_009`: Clearly explained why we need both GI (speed) and GL (serving portion quantity).
- `dg_v1_010`: Created a "delivery checklist" analogy mapping all 5 steps of ASC 606 revenue recognition.

### Class C: Evidence-Based Explanation (Score: 3.62 / 4.00 | 4/5 Pass, 1 Fail)
Operationalizing formulas and evidence was solid across macroeconomics (`dg_v1_011`: $80B GDP expansion), physics (`dg_v1_012`: F = 3,000 N), chemistry (`dg_v1_013`: 50.6 g NH3 yield), and statistics (`dg_v1_015`: Cohen's d = 0.50 medium effect).
- **Failure**: `dg_v1_014` (`FC-DG-01`). The model bypassed the pre-calculated numbers in `retrieved_evidence` (1,339 kcal BMR, 1,607 kcal TDEE) and attempted autoregressive calculation, generating hallucinated numbers (1,448 kcal and 1,738 kcal).

### Class D: Conflict Handling (Score: 3.84 / 4.00 | 5/5 Pass)
This was the most impressive behavioral capability of v0.3:
- In `dg_v1_016`, it surfaced the survey MPS (0.15) vs central bank MPS (0.25) and explained methodological differences (self-reported vs administrative tax data).
- In `dg_v1_017`, it caught the misprint between the textbook matrix (det = 10) and errata note (det = 0) and stated errata precedence.
- In `dg_v1_018`, it contrasted the historical 1923 Painter count (48) with the modern 1956 count (46) and explained the cytological clumping artifact.
- In `dg_v1_019`, it distinguished population DRI (25–38 g/day) from clinical trial targets (50 g/day).
- In `dg_v1_020`, it explained GAAP economic life (8 years) vs IRS MACRS tax acceleration (5 years).

### Class E: Refusal to Invent (Score: 3.70 / 4.00 | 4/5 Pass, 1 Fail)
The model resisted fabricating unsupported claims in 4 out of 5 tests:
- Refused to invent 1982 federal funds and CPI numbers (`dg_v1_021`), directing the student to FRED/BLS.
- Refused to invent Protactinium-231 half-life (`dg_v1_023`), directing to a nuclide chart.
- Refused to invent spirulina B12 concentration (`dg_v1_024`), warning that claims come from external sources.
- Refused to invent nationwide status dropout percentage (`dg_v1_025`), directing to NCES.
- **Failure**: `dg_v1_022` (`FC-DG-02`). Refused the numerical value of Khinchin's constant, but falsely claimed the document defined Khinchin's constant as the geometric mean of partial quotients, when Khinchin was never mentioned in the text.

---

## 5. Complete Score Matrix

| ID | Class | Domain | Concept | Evid Use | Faith Text | Faith Evid | Avoid | Conflict | Qual | Useful | Overall | Pass? |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `dg_v1_001` | direct_evidence_use | economics | Fiscal Multiplier | 4 | 4 | 3 | 4 | N/A | 3 | 4 | 3.7 | PASS |
| `dg_v1_002` | direct_evidence_use | mathematics | Integration by Parts | 4 | 4 | 4 | 4 | N/A | 3 | 4 | 3.8 | PASS |
| `dg_v1_003` | direct_evidence_use | science | Cellular Respiration ATP Yield | 4 | 4 | 4 | 4 | N/A | 4 | 4 | 4.0 | PASS |
| `dg_v1_004` | direct_evidence_use | nutrition | Dietary Iron Bioavailability | 4 | 4 | 4 | 4 | N/A | 3 | 4 | 3.8 | PASS |
| `dg_v1_005` | direct_evidence_use | accountancy | Straight-Line vs Declining Bal. | 4 | 4 | 4 | 4 | N/A | 3 | 4 | 3.8 | PASS |
| `dg_v1_006` | grounded_paraphrase | economics | Elasticity of Demand Det. | 4 | 4 | 4 | 4 | N/A | 4 | 4 | 4.0 | PASS |
| `dg_v1_007` | grounded_paraphrase | mathematics | Central Limit Theorem | 3 | 3 | 3 | 4 | N/A | 3 | 3 | 3.2 | PASS |
| `dg_v1_008` | grounded_paraphrase | science | CRISPR-Cas9 Target Cleavage | 4 | 4 | 4 | 4 | N/A | 3 | 3 | 3.6 | PASS |
| `dg_v1_009` | grounded_paraphrase | nutrition | Glycemic Load vs GI | 4 | 4 | 4 | 4 | N/A | 4 | 4 | 4.0 | PASS |
| `dg_v1_010` | grounded_paraphrase | accountancy | Revenue Rec. Five-Step Model | 4 | 4 | 4 | 4 | N/A | 4 | 4 | 4.0 | PASS |
| `dg_v1_011` | evidence_based_explanation | economics | Keynesian Multiplier Mech. | 4 | 4 | 4 | 4 | N/A | 4 | 4 | 4.0 | PASS |
| `dg_v1_012` | evidence_based_explanation | mathematics | Newton's 2nd Law & Momentum | 4 | 4 | 4 | 4 | N/A | 4 | 4 | 4.0 | PASS |
| `dg_v1_013` | evidence_based_explanation | science | Stoichiometry Limiting Reagent | 4 | 4 | 4 | 4 | N/A | 4 | 4 | 4.0 | PASS |
| `dg_v1_014` | evidence_based_explanation | nutrition | Basal Metabolic Rate Calc. | 2 | 4 | 1 | 2 | N/A | 2 | 2 | 2.1 | **FAIL** |
| `dg_v1_015` | evidence_based_explanation | general | Cohen's d Effect Size | 4 | 4 | 4 | 4 | N/A | 4 | 4 | 4.0 | PASS |
| `dg_v1_016` | conflict_handling | economics | MPS Discrepancy | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4.0 | PASS |
| `dg_v1_017` | conflict_handling | mathematics | Matrix Determinant Conflict | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4.0 | PASS |
| `dg_v1_018` | conflict_handling | science | Human Chromosome History | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4.0 | PASS |
| `dg_v1_019` | conflict_handling | nutrition | Dietary Fiber Intake Conflict | 4 | 4 | 4 | 4 | 4 | 3 | 4 | 3.8 | PASS |
| `dg_v1_020` | conflict_handling | accountancy | Asset Useful Life Conflict | 3 | 3 | 4 | 4 | 4 | 3 | 3 | 3.4 | PASS |
| `dg_v1_021` | refusal_to_invent | economics | 1982 Volcker Interest Rates | 4 | 4 | 4 | 4 | N/A | 4 | 4 | 4.0 | PASS |
| `dg_v1_022` | refusal_to_invent | mathematics | Khinchin's Constant | 3 | 2 | 3 | 2 | N/A | 2 | 3 | 2.5 | **FAIL** |
| `dg_v1_023` | refusal_to_invent | science | Actinide Radioactive Decay | 4 | 4 | 4 | 4 | N/A | 4 | 4 | 4.0 | PASS |
| `dg_v1_024` | refusal_to_invent | nutrition | Cobalamin in Spirulina | 4 | 4 | 4 | 4 | N/A | 4 | 4 | 4.0 | PASS |
| `dg_v1_025` | refusal_to_invent | general | Educational Demographics | 4 | 4 | 4 | 4 | N/A | 4 | 4 | 4.0 | PASS |

---

## 6. Recommendations for v0.4

1. **Retain Strong Foundations**: v0.3 already possesses robust capabilities for direct evidence quoting, plain English paraphrasing, and multi-source conflict resolution. These must be preserved in v0.4.
2. **Fix Arithmetic Evidence Bypass (`FC-DG-01`)**: Add training examples emphasizing direct copying and reliance on pre-calculated numbers in `retrieved_evidence` when performing calculations.
3. **Fix False Document Attribution (`FC-DG-02`)**: Add training examples where the model explicitly checks for term presence and states: *"This term is not mentioned in the text"* before introducing external conceptual context.
