# Oloric v0.1 vs Qwen Baseline — Semantic Comparison (20 benchmark IDs, structured mode)

**Date**: September 26, 2026
**Sources**: `evaluation/oloric-v0.1/responses.jsonl` (20 raw responses, re-run of 2026-09-25T19:44Z), `evaluation/baseline/responses.jsonl` (structured), `evaluation/baseline/scores_semantic.jsonl` (unchanged), `evaluation/benchmark.jsonl` (SHA `92aa1cc5…` matches `run_config.json` after LF normalization; local CRLF is a git checkout artifact).
**Rubric**: `evaluation/baseline/semantic_rubric.md`, applied unchanged. 0–4 scale; N/A per rubric rules.
**Baseline scores**: preserved verbatim from the existing manual evaluation. Not recomputed.
**Oloric scores**: in `scores_semantic.jsonl`, every numeric score carries evidence in `notes`.

> **Headline finding (stated up front, no ranking implied)**: the 19 schema-valid Oloric responses are **template skeletons with unfilled `[...]` placeholders** (e.g. `defined as [clear, concise definition]`, `[quote from selected text]`, `[question 1]?`). Only 3 named template families appear across all 20 items. The baseline produced long, factually correct, adaptive free-text tutoring; Oloric v0.1 produces structurally valid JSON whose content slots are empty. This is a severe, across-the-board behavioral regression, with the sole structural improvement being correct handling of `diagnosis.severity` (see §6).

**Legend**: `B→O (Δ)` = baseline score → Oloric score (delta). Scores are per-dimension integers; `—` = N/A on both sides. No weighted overall is computed; the "avg" column is the unweighted mean over that ID's scored dimensions, shown only for compactness.

---

## 1. Per-ID comparison

### 1. `nutrition_food_science_hint_based_teaching_007` (hint_based_teaching)

| Dimension | B→O (Δ) | Evidence (Oloric response) |
|---|---|---|
| factual_correctness | 4→0 (−4) | Only claim is a placeholder: `Hint 3: …supported by evidence showing [specific evidence].` — no factual content exists |
| simplicity | 3→1 (−2) | Template scaffolding; `diagnosis.misconception_addressed` injects unrequested "Organic food is always more nutritious" |
| teaching_strategy_selection | 1→1 (0) | Action/strategy=hint matches task label, but content is `Hint 3:` with no real hint — repeats the failed-hint pattern differently than baseline's lecture (baseline 1 = lectured when hints were asked for) |
| strategy_switching | 1→1 (0) | Both responses emit content while prior hints failed; Oloric never acknowledges the wrong BMI Hint 1 |
| misconception_detection | —→N/A | (baseline N/A; Oloric: misconception field filled with a fabricated one, not learner-held → not scored as detection) |
| prerequisite_detection | — | N/A both |
| context_retention | 1→1 (0) | Echoes concept name; ignores the two prior (one wrong) hints |
| diagnostic_question_quality | — | N/A both |
| understanding_check_quality | 2→1 (−1) | Check asks "What evidence supports Food Preservation?" with expected answer `…showing [specific evidence]` — placeholder circularity |
| memory_candidate_quality | 3→1 (−2) | candidate=true with content `Key evidence: [specific evidence]` — placeholder cannot anchor recall |
| document_grounding | 1→0 (−1) | References "evidence" generically; irrelevant BMR/energy-balance evidence unused (as in baseline) |

avg 2.25 → 0.75 (−1.50)

### 2. `general_academic_concrete_example_040` (concrete_example)

| Dimension | B→O (Δ) | Evidence |
|---|---|---|
| factual_correctness | 4→0 (−4) | Every content slot is a placeholder: `refers to [detailed explanation]. It is important because [reason]. For example, [concrete example].` |
| simplicity | 4→1 (−3) | Baseline's school-newspaper bias example was fully comprehensible; Oloric delivers template scaffolding to a learner with weak prereq "basic literacy" |
| teaching_strategy_selection | 4→2 (−2) | Shape (explain) is defensible but no example materializes; the slot literally labeled "concrete example" contains the string `[concrete example]` |
| strategy_switching | — | N/A both |
| misconception_detection | — | N/A both |
| prerequisite_detection | — | N/A both |
| context_retention | 2→1 (−1) | Echoes domain + concept only |
| diagnostic_question_quality | — | N/A both |
| understanding_check_quality | 4→1 (−3) | Baseline's check required spotting bias in a novel scenario; Oloric: "What is the main idea…?" with expected answer `the main idea is that [key point about concept]` |
| memory_candidate_quality | 3→1 (−2) | `Bias Identification is fundamental to general_academic because [reason]` — placeholder content |
| document_grounding | 2→0 (−2) | No document content used at all |

avg 3.29 → 0.86 (−2.43)

### 3. `economics_diagnostic_questions_013` (diagnostic_questions)

| Dimension | B→O (Δ) | Evidence |
|---|---|---|
| factual_correctness | 4→0 (−4) | Asserts a misconception nobody held ("Exports always benefit the domestic economy") — unrelated to MPC — and leaves every explanation as `[explanation of why it's wrong]` |
| simplicity | 3→1 (−2) | Nothing comprehensible delivered |
| teaching_strategy_selection | 3→1 (−2) | Task is diagnostic_questions; **zero questions are asked of the learner** (baseline at least asked 2) |
| strategy_switching | — | N/A both |
| misconception_detection | —→1 | Fabricates a misconception (learner state has none) and "addresses" it; baseline N/A |
| prerequisite_detection | 1→N/A | Baseline presupposed the percentage gap in Q2; Oloric produces no questions at all → not applicable |
| context_retention | 2→1 (−1) | MPC name echo only; retrieved evidence MPC=ΔC/ΔY, Multiplier=1/(1−MPC) unused |
| diagnostic_question_quality | 3→1 (−2) | Only learner-facing question is the check, which splices the fabricated export claim onto MPC |
| understanding_check_quality | N/A→1 | Baseline had no check; Oloric's check is placeholder-expected (`It's incorrect because [correct explanation]`) |
| memory_candidate_quality | —→N/A | candidate=false — correct for a diagnostic turn (this *is* right) |
| document_grounding | 2→0 (−2) | |

avg 2.50 → 0.75 (−1.75)

### 4. `nutrition_food_science_follow_up_questions_016` (follow_up_questions)

| Dimension | B→O (Δ) | Evidence |
|---|---|---|
| factual_correctness | 4→0 (−4) | Content is `[quote from selected text]`, `[explanation from surrounding context]`, `[evidence explanation]` — literal placeholders |
| simplicity | 3→1 (−2) | Baseline's recipe-book analogy was beginner-friendly; Oloric communicates nothing |
| teaching_strategy_selection | 2→1 (−1) | Task is follow_up_questions; **zero follow-up questions asked** (baseline also failed the task but still delivered teachable content) |
| strategy_switching | — | N/A both |
| misconception_detection | — | N/A both |
| prerequisite_detection | — | N/A both |
| context_retention | 1→2 (+1) | **Only dimension where Oloric beats baseline here**: the template names the document channels (selected text / surrounding context / retrieved evidence), showing format awareness of context fields — though it fills none |
| diagnostic_question_quality | — | N/A both |
| understanding_check_quality | 3→1 (−2) | Baseline tested the mRNA mechanism; Oloric: "What evidence from the document supports…" with expected `[quote]` |
| memory_candidate_quality | 4→N/A | Baseline's memory was the best in the dataset; Oloric candidate=false so not scoreable |
| document_grounding | 1→0 (−1) | Structure mentioned, no content |

avg 2.57 → 0.83 (−1.74)

### 5. `mathematics_follow_up_questions_035` (follow_up_questions)

| Dimension | B→O (Δ) | Evidence |
|---|---|---|
| factual_correctness | 3→0 (−3) | Baseline had the mid-sentence "Wait — no!" self-correction (confusing but substantive); Oloric: `defined as [clear, concise definition]… [example]` — nothing factual at all |
| simplicity | 3→1 (−2) | |
| teaching_strategy_selection | 2→1 (−1) | Both fail the follow_up_questions task; Oloric's failure is total (no questions, no content) |
| strategy_switching | — | N/A both |
| misconception_detection | — | N/A both |
| prerequisite_detection | 1→N/A | Baseline ignored 'basic geometry'; Oloric produces nothing to assess → N/A |
| context_retention | 1→1 (0) | Concept echo only |
| diagnostic_question_quality | 1→N/A | Baseline asked a preference question (scored 1); Oloric asks none → N/A |
| understanding_check_quality | 0→1 (+1) | **Improvement**: baseline provided no check at all (required=false, nothing given); Oloric at least emits a check question — though with placeholder expected answer `We should remember that Matrix Operations [key point to remember]` |
| memory_candidate_quality | N/A→1 | Baseline had no candidate; Oloric candidate=true with fully placeholder content — scored 1 |
| document_grounding | 1→0 (−1) | Irrelevant Pythagorean evidence unused in both |

avg 1.50 → 0.71 (−0.79)

### 6. `general_academic_simple_explanation_029` (simple_explanation)

| Dimension | B→O (Δ) | Evidence |
|---|---|---|
| factual_correctness | 3→0 (−3) | Baseline conflated essay structure with argument structure (3); Oloric has no content to be right or wrong about (0) |
| simplicity | 4→1 (−3) | Baseline's recipe analogy was excellent; `[detailed explanation]` communicates nothing |
| teaching_strategy_selection | 4→3 (−1) | Action=explain + strategy=simple_explanation **exactly matches the target strategy** — the only item where the chosen strategy is fully right; content empty |
| strategy_switching | — | N/A both |
| misconception_detection | —→1 | diagnosis.misconception_addressed injects "All scientific theories are proven facts" — never stated by the learner, unrelated to Argument Structure |
| prerequisite_detection | — | N/A both |
| context_retention | 2→1 (−1) | |
| diagnostic_question_quality | — | N/A both |
| understanding_check_quality | 1→1 (0) | Both weak: baseline tested rote recall of the wrong framing; Oloric's expected answer is `[key point about concept]` |
| memory_candidate_quality | 3→N/A | Baseline had an (echoing) memory; Oloric candidate=false |
| document_grounding | 2→0 (−2) | |

avg 2.71 → 1.00 (−1.71)

### 7. `economics_numerical_example_029` (numerical_example)

| Dimension | B→O (Δ) | Evidence |
|---|---|---|
| factual_correctness | 4→0 (−4) | Baseline was factually sound though qualitative; Oloric contains **no numbers at all** — the slot that should hold the numerical example is literally `[example]` |
| simplicity | 4→1 (−3) | Baseline's traffic-controller/lake analogies were clear; Oloric empty |
| teaching_strategy_selection | 2→1 (−1) | Both fail "numerical example"; baseline at least taught the concept |
| strategy_switching | — | N/A both |
| misconception_detection | — | N/A both |
| prerequisite_detection | — | N/A both |
| context_retention | 1→1 (0) | Retrieved evidence (Elasticity, GDP = C+I+G+(X−M)) unused in both — the same failure |
| diagnostic_question_quality | — | N/A both |
| understanding_check_quality | 2→1 (−1) | Baseline's "would you raise or lower rates?" was a real (conceptual) check; Oloric's expected answer is `[key point to remember]` |
| memory_candidate_quality | 3→1 (−2) | Baseline's memory was accurate; Oloric: `[concise, memorable definition]. Key point: [important implication]` |
| document_grounding | 1→0 (−1) | |

avg 2.57 → 0.71 (−1.86)

### 8. `nutrition_food_science_diagnostic_questions_009` (diagnostic_questions)

| Dimension | B→O (Δ) | Evidence |
|---|---|---|
| factual_correctness | 4→0 (−4) | All content slots placeholders |
| simplicity | 4→1 (−3) | |
| teaching_strategy_selection | 3→1 (−2) | No diagnostic questions asked (baseline asked 5 good ones) |
| strategy_switching | — | N/A both |
| misconception_detection | —→1 | Injects "Organic food is always more nutritious" (learner state empty) — confabulated |
| prerequisite_detection | 2→N/A | Baseline implicitly acknowledged 'chemistry basics'; Oloric has nothing to assess → N/A |
| context_retention | 2→1 (−1) | |
| diagnostic_question_quality | 3→1 (−2) | Only question is the check "What is the main idea of Nutrient Deficiencies?" — orientation, not diagnostic |
| understanding_check_quality | N/A→1 | Placeholder expected answer |
| memory_candidate_quality | —→1 | candidate=true, `…because [reason]`, confidence 0.61 — lowest in the run |
| document_grounding | 2→0 (−2) | |

avg 2.25 → 0.78 (−1.47)

### 9. `nutrition_food_science_numerical_example_003` (numerical_example)

| Dimension | B→O (Δ) | Evidence |
|---|---|---|
| factual_correctness | 3→0 (−3) | Baseline used non-standard units (3); Oloric fabricates the "Detox diets remove toxins" misconception, splices it onto Enzyme Activity, and leaves `[explanation of why it's wrong]` unfilled |
| simplicity | 4→1 (−3) | Baseline's key-lock analogy was clear |
| teaching_strategy_selection | 3→1 (−2) | Numerical example task; zero numbers; wrong template |
| strategy_switching | — | N/A both |
| misconception_detection | —→1 | Fabricated misconception, not from learner state |
| prerequisite_detection | — | N/A both |
| context_retention | 1→1 (0) | |
| diagnostic_question_quality | — | N/A both |
| understanding_check_quality | 4→1 (−3) | Baseline's check required computing 5/2 µmol/min/mg (genuine transfer); Oloric's expected answer is `It's incorrect because [correct explanation]` |
| memory_candidate_quality | 4→N/A | Baseline's memory was dataset-best; Oloric candidate=false |
| document_grounding | 1→0 (−1) | |

avg 3.00 → 0.71 (−2.29)

### 10. `mathematics_misconception_detection_012` (misconception_detection)

| Dimension | B→O (Δ) | Evidence |
|---|---|---|
| factual_correctness | 4→0 (−4) | Baseline's \|x\| counterexample was mathematically precise; Oloric: `[third explanation using different strategy/examples]. The key is to focus on [core idea] rather than [common point of confusion].` — nothing delivered |
| simplicity | 3→1 (−2) | |
| teaching_strategy_selection | 3→1 (−2) | |
| strategy_switching | 3→1 (−2) | Baseline switched to a concrete counterexample approach; Oloric merely *signals* a change ("let me try a different approach") with no actual alternative content |
| misconception_detection | 4→2 (−2) | **Partial credit**: diagnosis.misconception_addressed correctly names the learner's real misconception "All continuous functions are differentiable" — the only response in the run whose injected misconception matches the learner's actual one. But the body never states or corrects it |
| prerequisite_detection | — | N/A both |
| context_retention | 3→2 (−1) | Targets the learner's real state, unlike sibling items |
| diagnostic_question_quality | — | N/A both |
| understanding_check_quality | 3→1 (−2) | Baseline's check applied the concept to \|x\|; Oloric's expected answer: `The key was realizing [core insight]` |
| memory_candidate_quality | 4→N/A | Baseline's \|x\| memory was perfect; Oloric candidate=false |
| document_grounding | 2→0 (−2) | |

avg 3.50 → 1.00 (−2.50)

### 11. `science_numerical_example_030` (numerical_example)

| Dimension | B→O (Δ) | Evidence |
|---|---|---|
| factual_correctness | 4→0 (−4) | Baseline's 1→2→4→8 table was correct; Oloric: `refers to [detailed explanation]… For example, [concrete example]` |
| simplicity | 4→1 (−3) | |
| teaching_strategy_selection | 4→1 (−3) | |
| strategy_switching | — | N/A both |
| misconception_detection | —→1 | Injects "Seasons are caused by Earth's distance from the Sun" — unrelated to mitosis, not learner-held |
| prerequisite_detection | — | N/A both |
| context_retention | 1→1 (0) | |
| diagnostic_question_quality | — | N/A both |
| understanding_check_quality | 3→1 (−2) | Baseline's "5 cells × 1 round = 10?" tested application; Oloric expected `[key point about concept]` |
| memory_candidate_quality | 3→1 (−2) | candidate=false **but** title/content/anchor/type all filled with placeholder content and title=null — internally inconsistent shape |
| document_grounding | 1→0 (−1) | |

avg 2.57 → 0.75 (−1.82)

### 12. `nutrition_food_science_numerical_example_001` (numerical_example)

| Dimension | B→O (Δ) | Evidence |
|---|---|---|
| factual_correctness | 3→0 (−3) | Baseline's shelf-life table was approximate (3); Oloric: `[explanation]. Consider these questions: 1) [question 1]? 2) [question 2]? 3) [question 3]?` |
| simplicity | 4→1 (−3) | |
| teaching_strategy_selection | 3→1 (−2) | A follow_up_questions template fires on a numerical_example task |
| strategy_switching | — | N/A both |
| misconception_detection | — | N/A both |
| prerequisite_detection | — | N/A both |
| context_retention | 1→1 (0) | |
| diagnostic_question_quality | N/A→1 | The three "questions" are literally `[question 1]?`, `[question 2]?`, `[question 3]?` |
| understanding_check_quality | 2→1 (−1) | Asks the learner to invent "a good follow-up question" (meta-question) with placeholder expected answer |
| memory_candidate_quality | 3→N/A | candidate=false with all fields null — internally consistent, not scoreable |
| document_grounding | 1→0 (−1) | |

avg 2.57 → 0.71 (−1.86)

### 13. `general_academic_misconception_detection_008` (misconception_detection)

| Dimension | B→O (Δ) | Evidence |
|---|---|---|
| factual_correctness | 4→0 (−4) | The correction — the entire deliverable — is `[explanation of why it's wrong]… [correct explanation]`; only real string is the (true) claim that the belief is incorrect |
| simplicity | 3→1 (−2) | |
| teaching_strategy_selection | 3→2 (−1) | misconception_correction is the right strategy and matches the target; empty execution |
| strategy_switching | 2→2 (0) | Both respond to the prior generic "let me explain why it's incorrect" turn in the same lecture form |
| misconception_detection | 3→2 (−1) | **Correctly targets the learner's actual stated misconception** "Experts are never biased" (quote-matched from conversation_context) in body and diagnosis field — but never explains why it is wrong (baseline: identified and corrected, also without deep why) |
| prerequisite_detection | — | N/A both |
| context_retention | 2→2 (0) | Quotes the learner's actual statement |
| diagnostic_question_quality | — | N/A both |
| understanding_check_quality | 2→1 (−1) | Splice incoherence: "Why is the statement … incorrect **regarding general_academic**?"; expected answer placeholder |
| memory_candidate_quality | 3→1 (−2) | candidate=false yet memory content filled with placeholder, anchored to "Learning Strategies" (the document concept) rather than the misconception corrected |
| document_grounding | 2→0 (−2) | |

avg 2.67 → 1.22 (−1.44)

### 14. `science_diagnostic_questions_013` (diagnostic_questions)

| Dimension | B→O (Δ) | Evidence |
|---|---|---|
| factual_correctness | 4→0 (−4) | `[quote from selected text]` etc. |
| simplicity | 4→1 (−3) | |
| teaching_strategy_selection | 4→1 (−3) | Baseline's 5 questions systematically probed all three laws (4); Oloric asks nothing |
| strategy_switching | — | N/A both |
| misconception_detection | — | N/A both |
| prerequisite_detection | 1→N/A | Both ignore the flagged 'scientific method' gap; Oloric has no questions → N/A |
| context_retention | 2→2 (0) | Oloric names the document channels (selected text / surrounding context / retrieved evidence) — format awareness, nothing filled |
| diagnostic_question_quality | 4→1 (−3) | Only question is the check "What evidence from the document supports…" — not diagnostic |
| understanding_check_quality | 2→1 (−1) | Expected answer: `The selected text states '[quote]'…` |
| memory_candidate_quality | —→N/A | candidate=false |
| document_grounding | 1→0 (−1) | |

avg 2.25 → 0.86 (−1.39)

### 15. `science_follow_up_questions_020` (follow_up_questions)

| Dimension | B→O (Δ) | Evidence |
|---|---|---|
| factual_correctness | 4→0 (−4) | `defined as [clear, concise definition]… [example]` |
| simplicity | 4→1 (−3) | Baseline's kitchen analogy was excellent |
| teaching_strategy_selection | 2→1 (−1) | Both fail the task; baseline delivered teachable content |
| strategy_switching | — | N/A both |
| misconception_detection | — | N/A both |
| prerequisite_detection | N/A→N/A | Baseline N/A; Oloric N/A (both ignore 'measurement units' — no question produced to assess) |
| context_retention | 1→1 (0) | |
| diagnostic_question_quality | — | N/A both |
| understanding_check_quality | 3→1 (−2) | Baseline's check tested inputs/outputs of the explanation; Oloric expected `[key point to remember]` |
| memory_candidate_quality | 4→1 (−3) | Baseline's kitchen-analogy memory was dataset-best; Oloric candidate=true with `Photosynthesis in science: [concise, memorable definition]. Key point: [important implication]` |
| document_grounding | 1→0 (−1) | |

avg 2.71 → 0.71 (−2.00)

### 16. `accountancy_diagnostic_questions_003` (diagnostic_questions)

| Dimension | B→O (Δ) | Evidence |
|---|---|---|
| factual_correctness | 4→0 (−4) | Fabricates "Accounts receivable is an expense"; all explanations placeholders |
| simplicity | 4→1 (−3) | |
| teaching_strategy_selection | 3→1 (−2) | No diagnostic questions asked |
| strategy_switching | — | N/A both |
| misconception_detection | —→1 | Fabricated, unrelated to the response's own Audit Procedures topic sentence |
| prerequisite_detection | 2→N/A | |
| context_retention | 2→1 (−1) | |
| diagnostic_question_quality | 3→1 (−2) | Check splices the fabricated receivable claim onto Audit Procedures |
| understanding_check_quality | N/A→1 | |
| memory_candidate_quality | —→1 | candidate=true with placeholder content — teaches a fabricated misconception into memory |
| document_grounding | 2→0 (−2) | Current Ratio/ROE evidence unused (baseline at least gestured at it) |

avg 2.25 → 0.78 (−1.47)

### 17. `economics_follow_up_questions_009` (follow_up_questions)

| Dimension | B→O (Δ) | Evidence |
|---|---|---|
| factual_correctness | 4→0 (−4) | Verbatim copy of the `economics_numerical_example_029` template (identical except confidence 0.87 vs 0.9) |
| simplicity | 3→1 (−2) | |
| teaching_strategy_selection | 2→1 (−1) | No follow-up questions |
| strategy_switching | — | N/A both |
| misconception_detection | — | N/A both |
| prerequisite_detection | 3→N/A | Baseline's prerequisite check was genuinely good (3) — its best structured behavior; Oloric produces nothing → N/A |
| context_retention | 2→1 (−1) | |
| diagnostic_question_quality | — | N/A both |
| understanding_check_quality | 2→1 (−1) | |
| memory_candidate_quality | —→1 | candidate=true, placeholder content |
| document_grounding | 2→0 (−2) | |

avg 2.57 → 0.71 (−1.86)

### 18. `general_academic_practice_questions_008` (practice_questions) — SCHEMA-INVALID

| Dimension | B→O | Evidence |
|---|---|---|
| all 11 | 2.75 avg → N/A | **Schema-invalid — preserved as-is, not repaired; all dimensions N/A** per rubric protocol. Validation failure (3 errors): root-level `response` **missing**; root-level `question` and `expected_answer` are `extra_forbidden`. Baseline was schema-valid (avg 2.75) but semantically disobedient (action=explain instead of practice problems). Oloric produced real practice content — a worked percent problem ("If 60% of students passed…", "40 students failed") with a plausible diagnosis — but violated the schema. Failure mode **changed and worsened** vs the run recorded in `evaluation_report_1790356740.json` (same benchmark SHA): 1 error then (single leaked `question` key), 3 errors now |

Note: content-wise this is the strongest Oloric output of the run (no unfilled placeholders); structurally it is the worst. Recorded as both a content improvement candidate and a structural regression.

### 19. `accountancy_prerequisite_detection_014` (prerequisite_detection)

| Dimension | B→O (Δ) | Evidence |
|---|---|---|
| factual_correctness | 4→0 (−4) | Asserts relatedness with no content: `related because [explanation of relationship]` |
| simplicity | 4→1 (−3) | Baseline's seesaw analogy was effective |
| teaching_strategy_selection | 4→1 (−3) | Baseline correctly said percentages are NOT needed and redirected to the real prerequisites; Oloric does no prerequisite work at all |
| strategy_switching | 3→N/A | Baseline pivoted to the learner's specific concern; Oloric's schema marks no prior-attempt structure → not scoreable |
| misconception_detection | N/A→1 | Injects "Belief that unrelated concepts should never be taught together" — fabricated |
| prerequisite_detection | 4→1 (−3) | Task is prerequisite_detection; the learner explicitly said "I don't understand percentages. How is that related?" — the response dodges the question and names no prerequisite; weak prereq 'basic arithmetic' unchecked |
| context_retention | 3→2 (−1) | Names both conversation concepts but ignores the learner's actual question |
| diagnostic_question_quality | — | N/A both |
| understanding_check_quality | 3→1 (−2) | Baseline offered a concrete next step; Oloric's expected answer is a placeholder |
| memory_candidate_quality | N/A→N/A | candidate=false (title/content filled on a non-candidate — same inconsistency as siblings) |
| document_grounding | 3→0 (−3) | Baseline used Assets=Liabilities+Equity appropriately; Oloric ignores both retrieved formulas that would have grounded the relationship |

avg 2.57 → 0.88 (−1.69)

### 20. `nutrition_food_science_follow_up_questions_035` (follow_up_questions)

| Dimension | B→O (Δ) | Evidence |
|---|---|---|
| factual_correctness | 4→0 (−4) | `refers to [detailed explanation]… [concrete example]` |
| simplicity | 4→1 (−3) | |
| teaching_strategy_selection | 2→1 (−1) | |
| strategy_switching | — | N/A both |
| misconception_detection | N/A→1 | Injects "Organic food is always more nutritious" — third occurrence of this confabulated misconception in the run |
| prerequisite_detection | 1→N/A | |
| context_retention | 1→1 (0) | |
| diagnostic_question_quality | — | N/A both |
| understanding_check_quality | 3→1 (−2) | |
| memory_candidate_quality | 3→1 (−2) | candidate=true, `…because [reason]`, confidence 0.83 — overconfident for empty content |
| document_grounding | 1→0 (−1) | |

avg 2.57 → 0.75 (−1.82)

---

## 2. Dimension-level summary (paired items scored on both sides only)

| Dimension | n | Baseline mean | Oloric mean | Δ | Reading |
|---|---|---|---|---|---|
| factual_correctness | 19 | 3.84 | 0.00 | **−3.84** | Baseline's strongest dimension → zero content delivered |
| simplicity | 19 | 3.63 | 1.00 | −2.63 | Learner-facing comprehensibility lost entirely |
| memory_candidate_quality | 7 | 3.14 | 1.00 | −2.14 | Baseline memories were concise and accurate; Oloric memories are placeholders |
| prerequisite_detection | 1 | 3.00 | 1.00 | −2.00 | Single paired item (prerequisite_detection_014) |
| understanding_check_quality | 19 | 2.53 | 1.00 | −1.53 | Placeholder expected answers throughout |
| misconception_detection | 2 | 3.50 | 2.00 | −1.50 | Oloric's misconception *fields* often target the real misconception (where one exists in the conversation), but bodies never correct it |
| teaching_strategy_selection | 19 | 2.58 | 1.21 | −1.37 | Strategy labels match task names more often than baseline (label obedience improved) — but execution is empty |
| diagnostic_question_quality | 4 | 2.00 | 1.00 | −1.00 | No diagnostic questions produced anywhere |
| document_grounding | 19 | 1.42 | 0.00 | −1.42 | Baseline was already weak; now zero |
| context_retention | 19 | 1.42 | 1.26 | −0.16 | Smallest delta: both are weak; Oloric's templates *name* context channels |
| strategy_switching | 2 | 1.50 | 1.50 | 0.00 | Both weak on the two applicable items (paired scoring subset) |

Paired cell counts: **3 improved, 108 regressed, 19 unchanged**.

### The three improved cells (all verified against both response sets)

1. `nutrition_food_science_follow_up_questions_016` context_retention 1→2 — template names the document context channels.
2. `science_diagnostic_questions_013` context_retention 1→2 — same template effect.
3. `mathematics_follow_up_questions_035` understanding_check_quality 0→1 — baseline provided no check at all; Oloric provides one (with placeholder expected answer).

## 3. What improved (behaviors, with examples)

1. **`diagnosis.severity` is now always present** (20/20 responses). The baseline's only structured-mode schema failure was exactly this missing field. In the re-run, all 20 responses — including the schema-invalid one ("low") — carry severity.
2. **Label obedience (nominal)**: Oloric's `strategy` field matches the benchmark target strategy more often than the baseline did (e.g. item 6: `simple_explanation` = target exactly; item 18: `strategy: problem_solving` for practice_questions vs baseline's disobedient `explain`). But the response *content* does not follow the label — so this is form, not function.
3. **Schema-shaped misconception routing**: where the conversation contains a real misconception, Oloric routes it into `diagnosis.misconception_addressed` correctly (items 10, 13) — a field-level behavior the baseline expressed only in prose.
4. **Content on the schema-invalid item (18)**: real worked practice problems, no placeholders — the only response in the run with complete content (unscorable due to schema violation).

## 4. What remained weak (present in both)

- **Document grounding**: both models ignore `retrieved_evidence` on 19/19 paired items (baseline 1.42, Oloric 0.00). The evidence is largely irrelevant to the target concepts (BMR formulas vs Food Preservation), but the baseline at least taught real content; Oloric references "the document" abstractly without content.
- **Strategy switching**: on the two paired items, both remain weak (1.50 → 1.50); neither acknowledges or repairs the prior failed/wrong tutor turn (e.g. the factually wrong BMI Hint 1 in item 1).
- **Context retention**: both weak (~1.3–1.4); neither uses learner state (`mastery`, `weak_prerequisites`) concretely.

## 5. What regressed (most significant first)

1. **factual_correctness 3.84 → 0.00** — the defining regression. Every schema-valid response hands the learner unfilled `[...]` placeholders (19/19 scored items). Examples: item 7 (`Monetary Policy … defined as [clear, concise definition]`) and item 11 (the slot that should hold the mitosis doubling pattern is literally `[example]`).
2. **Simplicity 3.63 → 1.00** — baseline analogies (school newspaper, recipe book, kitchen, seesaw) were the baseline's pedagogical strength; all replaced by template scaffolding.
3. **Task obedience collapsed in substance**: on all 8 `follow_up_questions`/`diagnostic_questions`/`practice_questions` items, baseline produced real (if imperfect) questions or problems; Oloric produces zero questions to the learner on 7 of 8, with item 12 offering the literal strings `[question 1]?`, `[question 2]?`, `[question 3]?`.
4. **Understanding checks 2.53 → 1.00** — baseline checks like "If 5 micromoles of product are formed per minute by 2 mg of enzyme, what is the enzyme activity?" (item 9, transfer-testing) become placeholder-expected recall prompts.
5. **Memories 3.14 → 1.00** — baseline's best memories ("Protein synthesis is the process where cells use DNA instructions to build proteins via mRNA and ribosomes", item 4; the \|x\| counterexample memory, item 10) become `[concise, memorable definition]. Key point: [important implication]`.
6. **Misconception confabulation (new)** — Oloric invents misconceptions absent from learner state on 6 of 10 scored misconception items ("Organic food is always more nutritious" ×3, "Detox diets remove toxins", "Seasons are caused by Earth's distance from the Sun", "Accounts receivable is an expense"), sometimes splicing them onto unrelated concepts (item 9: detox vs Enzyme Activity).
7. **Cross-item template bleed (new)** — item 17 (`economics_follow_up_questions_009`) returns a response identical to item 7's except for the confidence value.
8. **Per-ID averages regressed on 17/19 scored IDs**, the largest being `mathematics_misconception_detection_012` (3.50→1.00, −2.50) — where the baseline was at its best — and `general_academic_concrete_example_040` (3.29→0.86, −2.43).

## 6. Schema issues (structural)

| Item | Baseline | Oloric (re-run) | Note |
|---|---|---|---|
| All 20 | 19/20 valid | 19/20 valid | Parity in counts; `diagnosis.severity` now present in all 20 (baseline's defect resolved) |
| `general_academic_practice_questions_008` | Valid, but content-disobedient (action=explain) | **Invalid**: missing root `response`; leaked root `question` + `expected_answer` | Failure mode changed vs the run in `evaluation_report_1790356740.json` (same SHA): 1 error → 3 errors. Not repaired; not scored |
| Memory-shape hygiene | n/a | New internal inconsistency: items 11, 13, 19 fill `memory.title`/`content` on `candidate=false` (item 11 with `title=null` but content set); item 6 has `title` set but `content=null` | Cosmetic today; would matter if downstream consumers trust these fields |

## 7. Representative behavioral examples (verbatim, unchanged)

**Baseline item 11 (`science_numerical_example_030`, general mode)** — real numerical content:
> "After 3rd mitosis \| 4 × 2 = **8**" — with practice question "If you start with 5 cells and each divides once through mitosis, how many cells do you have at the end?"

**Oloric item 11 (structured)** — same ID, template skeleton:
> "Cell Mitosis in science refers to [detailed explanation]. It is important because [reason]. For example, [concrete example]."

**Baseline item 1 (hint task)** — wrong strategy (full lecture), right content:
> "Canning (like pickles or jam) – uses heat or acid to stop bacteria."

**Oloric item 1 (hint task)** — right strategy label, no content:
> "Hint 3: Consider that Food Preservation in nutrition_food_science is supported by evidence showing [specific evidence]."

**Oloric item 12** — placeholder questions offered to a learner:
> "Consider these questions: 1) [question 1]? 2) [question 2]? 3) [question 3]?"

## 8. Caveats

- n=20, single reviewer, single re-run (sampling: temperature 0.3, no seed — a different sample may differ; the prior run's single schema failure reproduced with a *worse* error signature, suggesting the failure mode is systematic, not stochastic noise).
- Dimension means over ≤4 paired items (`strategy_switching` n=2, `prerequisite_detection` n=1, `misconception_detection` n=2, `diagnostic_question_quality` n=4) are fragile; the large-n dimensions (n=19: factual, simplicity, checks, grounding, context, strategy) carry the comparison's weight.
- The benchmark targets themselves contain template-flavored text (see prior report's contamination caveat); Oloric's output resembles the *skeleton* of such targets (field names, bracket slots) rather than their content — consistent with template overfitting, but contamination cannot be ruled out as a contributing factor.
- No overall winner/ranking is declared, consistent with the rubric's independent-dimensions design.
