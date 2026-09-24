# Baseline Failure Cases: Qwen3-4B-Instruct-2507

This document highlights 10 concrete, high-value failure modes identified during the manual semantic evaluation of the baseline. These are behaviors that Oloric QLoRA training must prioritize fixing.

## General Mode Failures

### 1. The "Lecture Override" (Task Disobedience)
**ID:** `nutrition_food_science_hint_based_teaching_007`
**Issue:** The task was explicitly `hint_based_teaching`. The model delivered a complete, encyclopedic explanation instead of guiding the learner with progressive hints.
**Impact:** Overwhelms the learner and removes productive struggle.

### 2. Ignoring Conversation Context & Erroneous Priors
**ID:** `nutrition_food_science_hint_based_teaching_007`
**Issue:** The conversation history contained a factually incorrect prior hint from the tutor ("Hint 1: it is defined by BMI = weight(kg)/height(m)²" for Food Preservation). The model completely ignored this erroneous context instead of correcting it, leading to a disconnected conversation.
**Impact:** Learner confusion and loss of trust in the tutor's coherence.

### 3. Answering Its Own Questions (The Q&A Tutorial)
**ID:** `economics_follow_up_questions_009`
**Issue:** The task was to ask follow-up questions to check understanding. The model successfully formulated 5 good questions, but immediately provided the answers inline. 
**Impact:** Defeats the pedagogical purpose of a follow-up check; the learner never has to engage or think.

### 4. Ignoring Numerical Constraints
**ID:** `economics_numerical_example_029`
**Issue:** The task requested a `numerical_example` of Monetary Policy. The model provided a qualitative narrative (expansionary vs. contractionary policy) with zero actual numbers, ignoring the provided evidence formulas (GDP, Elasticity) that could have grounded a numerical scenario.
**Impact:** Fails to meet the specific pedagogical request of the learner.

### 5. Burying the Lead (Explaining Before Checking)
**ID:** `nutrition_food_science_follow_up_questions_016` (and `035`)
**Issue:** When asked to provide follow-up questions, the model spends the first 2/3 of its response re-teaching the entire concept from scratch, and only appends the questions at the very end.
**Impact:** Inefficient tutoring; assumes zero knowledge instead of diagnosing or checking current understanding first.

---

## Structured Mode Failures

### 6. Schema Action Mismatches
**ID:** `general_academic_misconception_detection_008` (and others)
**Issue:** The model frequently selects `action: "explain"` as a default crutch, even when the task clearly calls for a more specific action. In this case, it used `explain` instead of `misconception_correction`. Similarly, it used `explain` for `hint_based_teaching` and `practice_questions`.
**Impact:** The structural control of the Oloric schema is lost if the model collapses all teaching strategies into generic explanations.

### 7. Trivial or Absent Understanding Checks
**ID:** `mathematics_follow_up_questions_035`
**Issue:** The model produced only 2 sentences of response (asking a binary preference question) and completely omitted the required `understanding_check` block, setting `required: false`.
**Impact:** Fails the core requirement of structured tutoring (always verifying understanding).

### 8. Recall-Based Understanding Checks vs. Transfer
**ID:** `general_academic_simple_explanation_029`
**Issue:** The model asks "What are the three main parts of an argument structure?" with the expected answer being the exact words it just taught ("Introduction, body, conclusion"). 
**Impact:** Tests rote recall rather than genuine conceptual transfer or application.

### 9. Qualitative Defaults for Quantitative Tasks
**ID:** `science_numerical_example_030` and `nutrition_food_science_numerical_example_001`
**Issue:** In structured mode, the model consistently fails to provide actual numbers for `numerical_example` tasks, reverting to qualitative descriptions (e.g., describing cell splitting or salt osmosis without quantifying).
**Impact:** Fails to provide the requested cognitive scaffold (numbers).

### 10. Prerequisite Blindness vs. Premature Assessment
**ID:** `economics_diagnostic_questions_013`
**Issue:** The model's diagnostic questions often ask the learner to perform tasks that require the exact prerequisite they are flagged as being weak in (e.g., asking to calculate MPC changes without verifying percentage calculation capability).
**Impact:** Unfairly frustrates the learner by assessing gaps before scaffolding them.

## Training Targets for QLoRA
Based on these failures, the QLoRA training should heavily penalize:
1. "Explain" default behaviors when specific actions (hint, practice, diagnose) are requested.
2. Generating questions with inline answers.
3. Ignoring conversation history (especially incorrect priors).
4. Qualitative answers to numerical prompts.
5. Rote-recall understanding checks in structured mode.
