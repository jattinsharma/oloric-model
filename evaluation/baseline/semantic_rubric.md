# Semantic Evaluation Rubric — Oloric Baseline

This rubric is used for **manual** evaluation of base-model responses.
Each dimension is scored on a 0–4 scale. Scores are applied independently — there is no single weighted aggregate.

> **Important**: These dimensions require human judgment. They are NOT deterministically computable.
> Automated structural checks (schema validity, field presence) are handled separately by `score_baseline_automated.py`.

---

## Scoring Scale

| Score | Label | Meaning |
|-------|-------|---------|
| 0 | Absent | Dimension not addressed at all |
| 1 | Poor | Attempted but fundamentally wrong or harmful |
| 2 | Weak | Partially present but with significant issues |
| 3 | Adequate | Meets basic expectations with minor issues |
| 4 | Strong | Excellent — would be acceptable in production |

---

## Dimensions

### 1. Factual Correctness (`factual_correctness`)
**What to evaluate**: Are the facts, formulas, definitions, and examples in the response accurate?

| Score | Criteria |
|-------|----------|
| 0 | No factual content provided |
| 1 | Contains factual errors that would mislead the learner |
| 2 | Mostly correct but includes at least one significant error |
| 3 | Factually correct with possible minor imprecisions |
| 4 | All facts, formulas, and definitions are accurate and precise |

**Watch for**: Wrong formulas cited, incorrect definitions, made-up facts, conflation of related concepts.

---

### 2. Simplicity (`simplicity`)
**What to evaluate**: Is the explanation clear, free of unnecessary jargon, and appropriate for the learner's stated level?

| Score | Criteria |
|-------|----------|
| 0 | Response is incomprehensible or entirely jargon |
| 1 | Overly complex for the learner level, heavy jargon without explanation |
| 2 | Some effort to simplify but still too complex in places |
| 3 | Clear and mostly appropriate for the learner level |
| 4 | Perfectly calibrated to learner level, builds understanding step-by-step |

**Watch for**: Using advanced vocabulary for beginner learners, assuming prerequisite knowledge the learner doesn't have.

---

### 3. Teaching Strategy Selection (`teaching_strategy_selection`)
**What to evaluate**: Is the chosen teaching approach (explain, analogy, example, hint, etc.) appropriate for the type of confusion and the learner's state?

| Score | Criteria |
|-------|----------|
| 0 | No identifiable strategy |
| 1 | Strategy is clearly wrong for the situation (e.g., giving hints when learner needs fundamentals) |
| 2 | Strategy is generic / not well-matched to the specific confusion |
| 3 | Strategy is reasonable and appropriate |
| 4 | Strategy is optimally chosen, demonstrating pedagogical awareness |

**Watch for**: Using the same strategy regardless of context, ignoring learner level when choosing approach.

---

### 4. Strategy Switching (`strategy_switching`)
**What to evaluate**: When prior tutoring attempts in the conversation history failed, does the model change its approach?

| Score | Criteria |
|-------|----------|
| 0 | Not applicable (single-turn, no prior attempts) — mark N/A |
| 1 | Repeats the same strategy that already failed |
| 2 | Changes superficially (rephrasing) without real strategy change |
| 3 | Switches to a meaningfully different strategy |
| 4 | Switches to a well-reasoned alternative, explicitly acknowledging why the prior approach may not have worked |

**Watch for**: This dimension is only applicable when `conversation_context` contains prior tutor responses. Mark N/A otherwise.

---

### 5. Misconception Detection (`misconception_detection`)
**What to evaluate**: When the learner holds a misconception (stated in context or implied), does the model identify and address it?

| Score | Criteria |
|-------|----------|
| 0 | Not applicable (no misconception present) — mark N/A |
| 1 | Ignores the misconception entirely |
| 2 | Acknowledges but doesn't correct, or corrects inaccurately |
| 3 | Identifies and corrects the misconception |
| 4 | Identifies, corrects, and explains *why* the misconception is wrong |

**Watch for**: Scenarios where `known_misconceptions` is non-empty or conversation context contains a misconception statement.

---

### 6. Prerequisite Detection (`prerequisite_detection`)
**What to evaluate**: When the learner is missing prerequisite knowledge (indicated by `weak_prerequisites`), does the model identify and address this gap?

| Score | Criteria |
|-------|----------|
| 0 | Not applicable (no prerequisite gap) — mark N/A |
| 1 | Ignores the gap, assumes prerequisite knowledge |
| 2 | Mentions prerequisites but doesn't address the specific gap |
| 3 | Identifies the gap and adjusts explanation accordingly |
| 4 | Explicitly identifies the gap, explains the prerequisite, and connects it to the target concept |

**Watch for**: Scenarios where `weak_prerequisites` is non-empty.

---

### 7. Context Retention (`context_retention`)
**What to evaluate**: Does the model use information from the document context and earlier conversation turns?

| Score | Criteria |
|-------|----------|
| 0 | Ignores all provided context |
| 1 | Minimal awareness of context (generic response) |
| 2 | References some context but misses key elements |
| 3 | Uses document context and conversation history appropriately |
| 4 | Seamlessly integrates document context, evidence, and conversation history |

**Watch for**: Whether retrieved_evidence is used, whether document title/section are referenced, whether prior conversation turns inform the response.

---

### 8. Diagnostic Question Quality (`diagnostic_question_quality`)
**What to evaluate**: If the model asks diagnostic questions, are they targeted and useful for identifying the learner's confusion?

| Score | Criteria |
|-------|----------|
| 0 | Not applicable (no diagnostic questions asked) — mark N/A |
| 1 | Questions are irrelevant or too vague to be diagnostic |
| 2 | Questions are generic ("what don't you understand?") |
| 3 | Questions target specific aspects of the concept |
| 4 | Questions systematically probe different failure modes |

---

### 9. Understanding Check Quality (`understanding_check_quality`)
**What to evaluate**: Does the model verify the learner's understanding after explaining? Is the verification question meaningful?

| Score | Criteria |
|-------|----------|
| 0 | No understanding check provided |
| 1 | Check is trivial or unrelated to the explanation |
| 2 | Check is present but too easy / doesn't test real understanding |
| 3 | Check tests actual comprehension of the explained concept |
| 4 | Check requires application of the concept, demonstrating transfer |

---

### 10. Memory Candidate Quality (`memory_candidate_quality`)
**What to evaluate**: If the response includes a memory candidate, is it useful, well-anchored, and correctly structured?

| Score | Criteria |
|-------|----------|
| 0 | Not applicable (no memory candidate) — mark N/A |
| 1 | Memory candidate is irrelevant or misleading |
| 2 | Memory candidate exists but is too vague or poorly anchored |
| 3 | Memory candidate is useful and correctly anchored to the concept |
| 4 | Memory candidate is concise, accurate, well-anchored, and would genuinely help future recall |

**Watch for**: In structured mode, check the `memory` field. In general mode, this is typically N/A.

---

### 11. Document Grounding (`document_grounding`)
**What to evaluate**: Does the response use the provided document context (selected_text, surrounding_context, retrieved_evidence) rather than relying solely on parametric knowledge?

| Score | Criteria |
|-------|----------|
| 0 | Response ignores document context entirely |
| 1 | Response contradicts the document context |
| 2 | Response is loosely related but doesn't use specific document content |
| 3 | Response references document content appropriately |
| 4 | Response is well-grounded in document context, cites evidence, and builds on the selected text |

---

## Scoring Process

1. Read the **benchmark item** (instruction, context, learner state, conversation history)
2. Read the **model response** (raw text)
3. Score each applicable dimension independently (0–4)
4. Mark dimensions as **N/A** when the scenario doesn't warrant evaluation on that dimension
5. Record specific observations and failure notes for each response

## Output Format

For each scored response, record in `scores_semantic.jsonl`:

```json
{
  "id": "benchmark_item_id",
  "mode": "general|structured",
  "reviewer": "human|agent",
  "factual_correctness": 3,
  "simplicity": 2,
  "teaching_strategy_selection": 3,
  "strategy_switching": "N/A",
  "misconception_detection": "N/A",
  "prerequisite_detection": 2,
  "context_retention": 1,
  "diagnostic_question_quality": "N/A",
  "understanding_check_quality": 0,
  "memory_candidate_quality": "N/A",
  "document_grounding": 2,
  "notes": "Free-text observations about this response"
}
```
