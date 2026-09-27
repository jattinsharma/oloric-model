#!/usr/bin/env python3
"""
OLORIC v0.4 Dataset Generator

Generates exactly 600 records according to the approved v0.4 architecture:
- 20 canonical categories from VALID_CATEGORIES (no new top-level categories)
- Core: 20 canonical categories * 24 records = 480 records
- Targeted:
  - 24 prior-error-correction records (category = "error_correction")
  - 96 document-grounded records (category = "document_grounded") across 5 behaviors:
    A. Direct evidence use
    B. Grounded paraphrase
    C. Evidence-based explanation & prioritization (Pattern A & Pattern B)
    D. Conflict handling
    E. Refusal to invent / document absence
- Category counts:
  - document_grounded = 120
  - error_correction = 48
  - all other 18 canonical categories = 24 each
- Domain balance:
  - Exactly 6 domains * 100 records = 600 records
- Concept coverage:
  - Exactly 60 concepts * 10 records = 600 records
  - Formula: concept_idx = (track_index * 4 + i) % len(concepts)
  - Full direct training coverage for all 20 benchmark concepts
- Memory hygiene:
  - If candidate == False: title=None, content=None, anchor_concept=None, memory_type=None, confidence=0.0
- Deterministic with seed=42:
  - train = 480, validation = 60, test = 60
"""
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from v04_concept_registry import (
    VALID_CATEGORIES,
    VALID_DOMAINS,
    CONCEPT_REGISTRY,
    get_domain_concepts,
    get_concept_info,
    get_prior_error_scenario,
    get_evidence_data,
    get_absence_refusal,
    get_grounded_paraphrase,
    get_conflict_handling,
)

SEED = 42
TOTAL_RECORDS = 600
TRAIN_COUNT = 480
VAL_COUNT = 60
TEST_COUNT = 60

# ============================================================
# ID GENERATION
# ============================================================

def generate_example_id(category: str, domain: str, seq: int) -> str:
    """Generate deterministic v0.4 ID: v04_{domain}_{category}_{seq:03d}."""
    return f"v04_{domain}_{category}_{seq:03d}"


# ============================================================
# MEMORY HYGIENE
# ============================================================

def _make_memory(
    candidate: bool,
    title: Optional[str] = None,
    content: Optional[str] = None,
    anchor_concept: Optional[str] = None,
    memory_type: Optional[str] = None,
    confidence: float = 0.0,
) -> Dict[str, Any]:
    """Construct memory object conforming strictly to schema hygiene."""
    if candidate:
        assert title is not None, "title required when candidate=True"
        assert content is not None, "content required when candidate=True"
        assert anchor_concept is not None, "anchor_concept required when candidate=True"
        assert memory_type is not None, "memory_type required when candidate=True"
        assert confidence > 0.0, "confidence > 0.0 required when candidate=True"
        return {
            "candidate": True,
            "title": title,
            "content": content,
            "anchor_concept": anchor_concept,
            "memory_type": memory_type,
            "confidence": confidence,
        }
    return {
        "candidate": False,
        "title": None,
        "content": None,
        "anchor_concept": None,
        "memory_type": None,
        "confidence": 0.0,
    }


# ============================================================
# CONTEXT BUILDER
# ============================================================

def create_base_context(
    domain: str,
    concept: str,
    info: Dict[str, Any],
    rng: random.Random,
    level: str = "beginner",
    conversation_turns: Optional[List[Dict[str, str]]] = None,
    task_type: str = "resolve_confusion",
    custom_doc_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create standardized context for an example using per-concept content."""
    if custom_doc_context:
        doc_context = custom_doc_context
    else:
        doc_evidence = info.get("document_evidence", [])
        selected_text = doc_evidence[0] if doc_evidence else f"The concept of {concept} is fundamental to understanding {domain}."
        surrounding = doc_evidence[1] if len(doc_evidence) > 1 else f"In the study of {domain}, {concept} plays a crucial role."
        formulas = info.get("formulas", [])
        retrieved = rng.sample(formulas, min(2, len(formulas))) if formulas else [f"Key principle of {concept}"]

        doc_context = {
            "document_id": f"{domain}_{concept.lower().replace(' ', '_').replace('(', '').replace(')', '')[:20]}",
            "title": f"Introduction to {concept}",
            "page": rng.randint(10, 200),
            "section": concept.split("(")[0].strip() if "(" in concept else concept,
            "selected_text": selected_text,
            "surrounding_context": surrounding,
            "retrieved_evidence": retrieved,
        }

    if conversation_turns is None:
        conversation_turns = [
            {"role": "student", "content": f"I'm struggling to understand {concept}."}
        ]

    prereqs = info.get("prerequisites", [])
    known_prereqs = rng.sample(prereqs, min(2, len(prereqs))) if prereqs else []
    weak_prereqs = rng.sample(prereqs, min(1, len(prereqs))) if prereqs else []

    return {
        "task": task_type,
        "document_context": doc_context,
        "learner_state": {
            "level": level,
            "concept": concept,
            "mastery": round(rng.uniform(0.15, 0.55), 2),
            "known_prerequisites": known_prereqs,
            "weak_prerequisites": weak_prereqs,
            "known_misconceptions": [],
        },
        "conversation_context": conversation_turns,
        "current_goal": f"Understand {concept} well enough to apply it in {domain} contexts",
    }


def _make_record(
    category: str,
    domain: str,
    concept: str,
    seq: int,
    info: Dict[str, Any],
    rng: random.Random,
    instruction: str,
    conversation_turns: List[Dict[str, str]],
    target: Dict[str, Any],
    task_type: str = "resolve_confusion",
    level: str = "beginner",
    custom_doc_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a standardized record with all required fields."""
    return {
        "id": generate_example_id(category, domain, seq),
        "category": category,
        "domain": domain,
        "instruction": instruction,
        "context": create_base_context(
            domain, concept, info, rng, level, conversation_turns, task_type, custom_doc_context
        ),
        "target": target,
    }


# ============================================================
# CATEGORY-SPECIFIC STUDENT TURNS
# ============================================================

def _get_student_turns(category: str, concept: str, info: Dict[str, Any], rng: random.Random) -> List[Dict[str, str]]:
    """Generate category-appropriate student conversation turns."""
    misconceptions = info.get("misconceptions", [])
    misc_text = misconceptions[0][0] if misconceptions else "a common assumption"

    templates = {
        "simple_explanation": [
            [{"role": "student", "content": f"I'm struggling to understand {concept}. Can you explain it simply?"}],
            [{"role": "student", "content": f"What exactly is {concept}? I've read the definition but it doesn't make sense to me."}],
            [{"role": "student", "content": f"I need a clear explanation of {concept}. The textbook is confusing."}],
            [{"role": "student", "content": f"Could you break down {concept} for me? I'm finding it really hard."}],
        ],
        "simplification": [
            [{"role": "student", "content": f"The definition of {concept} is way too complicated. Can you make it simpler?"}],
            [{"role": "student", "content": f"Explain {concept} like I'm a complete beginner, please."}],
            [{"role": "student", "content": f"I don't need all the technical jargon about {concept}. Just give me the plain-English version."}],
            [{"role": "student", "content": f"Can you simplify {concept}? I'm getting lost in all the details."}],
        ],
        "analogy": [
            [{"role": "student", "content": f"Can you give me an everyday analogy for {concept}? I learn better that way."}],
            [{"role": "student", "content": f"Is there something from everyday life that works like {concept}?"}],
            [{"role": "student", "content": f"I need a metaphor or analogy to help me picture {concept}."}],
            [{"role": "student", "content": f"What's a good real-world comparison for {concept}?"}],
        ],
        "concrete_example": [
            [{"role": "student", "content": f"Can you give me a real-world example of {concept} in action?"}],
            [{"role": "student", "content": f"I understand the theory of {concept}, but what does it look like in practice?"}],
            [{"role": "student", "content": f"Show me a specific situation where {concept} happens."}],
            [{"role": "student", "content": f"Can you walk me through a concrete scenario involving {concept}?"}],
        ],
        "numerical_example": [
            [{"role": "student", "content": f"Can you show me a worked numerical example of {concept}?"}],
            [{"role": "student", "content": f"I need to see the numbers. How do you calculate or quantify {concept}?"}],
            [{"role": "student", "content": f"Walk me through a step-by-step calculation showing {concept} with actual numbers."}],
            [{"role": "student", "content": f"Can we do a math problem that demonstrates {concept}?"}],
        ],
        "diagnostic_questions": [
            [{"role": "student", "content": f"I don't even know what I don't know about {concept}. Can you test where my gaps are?"}],
            [{"role": "student", "content": f"I'm confused about {concept} but can't put my finger on why."}],
            [{"role": "student", "content": f"Can you ask me some questions to figure out where I'm going wrong with {concept}?"}],
            [{"role": "student", "content": f"I keep failing quizzes on {concept}. Help me figure out what I'm missing."}],
        ],
        "follow_up_questions": [
            [{"role": "student", "content": f"I understand the basics of {concept} now. What should I think about next?"}],
            [{"role": "student", "content": f"I get the definition of {concept}. What deeper questions should I be asking?"}],
            [{"role": "student", "content": f"Now that I know what {concept} is, how do I push my understanding further?"}],
            [{"role": "student", "content": f"I've got the foundation of {concept}. Give me some challenging questions to consider."}],
        ],
        "prerequisite_detection": [
            [{"role": "student", "content": f"I can't seem to follow the lesson on {concept}. What should I review first?"}],
            [{"role": "student", "content": f"Is there something I need to know before tackling {concept}? I feel underprepared."}],
            [{"role": "student", "content": f"Why is {concept} so hard for me? Am I missing some background knowledge?"}],
            [{"role": "student", "content": f"What foundational topics should I master before studying {concept}?"}],
        ],
        "misconception_detection": [
            [{"role": "student", "content": f"Isn't it true that {misc_text} for {concept}?"}],
            [{"role": "student", "content": f"I was told that {misc_text}. Is that correct for {concept}?"}],
            [{"role": "student", "content": f"My classmate said that {misc_text}. Doesn't that make sense for {concept}?"}],
            [{"role": "student", "content": f"I've always assumed {misc_text} when thinking about {concept}. Am I right?"}],
        ],
        "multi_turn_tutoring": [
            [{"role": "student", "content": f"Can you explain {concept}?"},
             {"role": "tutor", "content": f"{concept} is fundamentally about {info['explanation']}."},
             {"role": "student", "content": f"OK, but why does that matter in practice?"}],
            [{"role": "student", "content": f"What's the purpose of {concept}?"},
             {"role": "tutor", "content": f"It helps us understand {info['reason']}."},
             {"role": "student", "content": f"Can you show me how it works step-by-step?"}],
            [{"role": "student", "content": f"I'm reviewing {concept} for an exam."},
             {"role": "tutor", "content": f"Let's make sure you have the key points down."},
             {"role": "student", "content": f"What's the most important thing to remember about it?"}],
            [{"role": "student", "content": f"How is {concept} used in the real world?"},
             {"role": "tutor", "content": f"It's applied in many contexts to {info['reason']}."},
             {"role": "student", "content": f"Can you give me a specific scenario?"}],
        ],
        "repeated_confusion": [
            [{"role": "student", "content": f"You explained {concept} before, but I still don't get it."},
             {"role": "tutor", "content": f"No problem at all. Let's look at {concept} from a different perspective."},
             {"role": "student", "content": f"I'm still confused. Can you try yet another way?"}],
            [{"role": "student", "content": f"We've been over {concept} twice and I'm still lost."},
             {"role": "tutor", "content": f"Let me try a completely different angle."},
             {"role": "student", "content": f"I appreciate your patience. Nothing is sticking yet."}],
            [{"role": "student", "content": f"I've read the chapter on {concept} three times and I still can't grasp it."},
             {"role": "tutor", "content": f"That's OK. Some concepts take multiple approaches."},
             {"role": "student", "content": f"Can you explain it in a way I haven't seen before?"}],
            [{"role": "student", "content": f"I tried your explanation of {concept} but it didn't help."},
             {"role": "tutor", "content": f"Let me try a different strategy."},
             {"role": "student", "content": f"Please try something very different from what we've done."}],
        ],
        "strategy_switching": [
            [{"role": "student", "content": f"I don't understand {concept} even after the explanation."},
             {"role": "tutor", "content": f"Let me try explaining {concept} differently."}],
            [{"role": "student", "content": f"The formal explanation of {concept} isn't working for me."},
             {"role": "tutor", "content": f"Let me switch to a different teaching approach."}],
            [{"role": "student", "content": f"I understood the analogy for {concept} but now I need the real explanation."},
             {"role": "tutor", "content": f"Good idea. Let me switch from analogy to concrete examples."}],
            [{"role": "student", "content": f"The textbook approach to {concept} is too abstract for me."},
             {"role": "tutor", "content": f"Let me try a more hands-on approach."}],
        ],
        "hint_based_teaching": [
            [{"role": "student", "content": f"I'm struggling with {concept}. Can you give me a hint instead of the answer?"},
             {"role": "tutor", "content": f"Sure! Here's your first hint: {info['hints'][0] if info.get('hints') else 'Think about the basics.'}"},
             {"role": "student", "content": f"I think I see it... can I have another hint?"}],
            [{"role": "student", "content": f"Don't tell me the answer to this {concept} problem. Just give me hints."},
             {"role": "tutor", "content": f"I like your approach! Let me guide you."},
             {"role": "student", "content": f"OK, what should I think about first?"}],
            [{"role": "student", "content": f"I want to figure out {concept} myself. Can you just point me in the right direction?"},
             {"role": "tutor", "content": f"Absolutely. Let me give you some hints."},
             {"role": "student", "content": f"I'm listening."}],
            [{"role": "student", "content": f"Give me clues about {concept} so I can work it out on my own."},
             {"role": "tutor", "content": f"Great learning strategy! Here's a starting hint."},
             {"role": "student", "content": f"That helps a bit. What else should I consider?"}],
        ],
        "practice_questions": [
            [{"role": "student", "content": f"I'd like to practice {concept} with some problems."},
             {"role": "tutor", "content": f"Great idea! Let's work through some exercises together."}],
            [{"role": "student", "content": f"I think I understand {concept} now. Can you give me practice problems to test myself?"},
             {"role": "tutor", "content": f"Absolutely. Here are some problems at increasing difficulty."}],
            [{"role": "student", "content": f"I have an exam on {concept} next week. Can we do practice questions?"},
             {"role": "tutor", "content": f"Let's prepare you with some targeted practice."}],
            [{"role": "student", "content": f"I want to make sure I can apply {concept}. Give me exercises."},
             {"role": "tutor", "content": f"Here are some problems. Try them and I'll check your work."}],
        ],
        "error_correction": [
            [{"role": "student", "content": f"I tried solving this {concept} problem but I think I got it wrong."},
             {"role": "tutor", "content": f"Let me look at your work and identify where you went off track."}],
            [{"role": "student", "content": f"I think {concept} means {misc_text}."},
             {"role": "tutor", "content": f"I notice there's a misunderstanding in your reasoning. Let me help correct it."}],
            [{"role": "student", "content": f"For {concept}, shouldn't you always assume {misc_text}?"},
             {"role": "tutor", "content": f"That's a very common mistake. Let's look at why that's incorrect."}],
            [{"role": "student", "content": f"I applied {concept} by doing the opposite of what makes sense."},
             {"role": "tutor", "content": f"Let's trace your steps and find the exact point of confusion."}],
        ],
        "partial_understanding": [
            [{"role": "student", "content": f"I understand part of {concept} — that {info['explanation'][:40]}... but I don't see the rest."}],
            [{"role": "student", "content": f"I get the basic idea of {concept}, but I don't understand why {info['reason'][:40]}..."}],
            [{"role": "student", "content": f"I know the definition of {concept}, but how does it connect to everything else?"}],
            [{"role": "student", "content": f"I understand what {concept} does, but not how or why."}],
        ],
        "understanding_confirmation": [
            [{"role": "student", "content": f"So {concept} means {info['explanation']}, right? Did I get that right?"}],
            [{"role": "student", "content": f"Let me check if I understand {concept}: it's essentially {info['explanation'][:50]}... Is that accurate?"}],
            [{"role": "student", "content": f"I think I've got it now. {concept} is important because {info['reason'][:50]}... Am I on track?"}],
            [{"role": "student", "content": f"Can I summarize {concept} in my own words and have you verify if I'm correct?"}],
        ],
        "memory_generation": [
            [{"role": "student", "content": f"Can you summarize the key points about {concept} so I can review them later?"}],
            [{"role": "student", "content": f"What are the absolute essentials I need to remember about {concept} for the exam?"}],
            [{"role": "student", "content": f"Give me a quick reference summary of {concept} that I can save."}],
            [{"role": "student", "content": f"Can you create a study-card summary of {concept} with the key takeaway?"}],
        ],
        "document_grounded": [
            [{"role": "student", "content": f"What does the textbook say about {concept}? Can you cite the specific evidence?"}],
            [{"role": "student", "content": f"Based on the reading material provided, how is {concept} defined and explained?"}],
            [{"role": "student", "content": f"Can you show me the exact passage and evidence that explains {concept}?"}],
            [{"role": "student", "content": f"According to the document excerpt, what are the key facts about {concept}?"}],
        ],
    }

    if category == "context_retention":
        domain = info.get("domain", "general_academic")
        other_concepts = [c for c in get_domain_concepts(domain) if c != concept]
        other = rng.choice(other_concepts) if other_concepts else concept
        return [
            {"role": "student", "content": f"Earlier we talked about {other}. Now I'm confused about {concept}."},
            {"role": "tutor", "content": f"Let's connect {other} to {concept}."},
        ]

    cat_templates = templates.get(category, templates["simple_explanation"])
    return rng.choice(cat_templates) if cat_templates else [{"role": "student", "content": f"I'm struggling to understand {concept}."}]


# ============================================================
# CORE GENERATORS (TRACKS 0 TO 19)
# ============================================================

def create_simple_explanation(domain, concept, info, seq, i, rng):
    turns = _get_student_turns("simple_explanation", concept, info, rng)
    ex_idx = i % len(info["concrete_examples"])
    example = info["concrete_examples"][ex_idx]
    return _make_record(
        "simple_explanation", domain, concept, seq, info, rng,
        instruction=f"Explain the concept of {concept} in {domain} clearly and simply.",
        conversation_turns=turns,
        target={
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "simple_explanation",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"To understand {concept} simply: it refers to {info['explanation']}. Understanding this matters because {info['reason']}. For instance, consider this concrete application: {example}",
            "understanding_check": {
                "required": True,
                "question": info["diagnostic_questions"][0] if info.get("diagnostic_questions") else f"What is the main idea of {concept}?",
                "expected_answer": f"The main idea is that {info['explanation']}."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": rng.choice(["low", "medium"]),
                "misconception_addressed": None,
            },
            "memory": _make_memory(
                candidate=rng.random() > 0.7,
                title=f"Key Concept: {concept}",
                content=f"{concept}: {info['explanation']}.",
                anchor_concept=concept,
                memory_type="definition",
                confidence=round(rng.uniform(0.6, 0.9), 2),
            ),
        },
    )


def create_simplification(domain, concept, info, seq, i, rng):
    turns = _get_student_turns("simplification", concept, info, rng)
    analogy = info["analogies"][0] if info.get("analogies") else f"Think of {concept} like an everyday process"
    return _make_record(
        "simplification", domain, concept, seq, info, rng,
        instruction=f"Simplify the explanation of {concept} for a beginner learner.",
        conversation_turns=turns,
        target={
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "simplification",
            "difficulty": "beginner",
            "response": f"To break down {concept} to its core principles: {analogy}. In practical terms, {concept} means {info['explanation']}. The essential insight to keep in mind is that {info['reason']}.",
            "understanding_check": {
                "required": True,
                "question": f"In your own words, what does {concept} mean?",
                "expected_answer": f"It means {info['explanation']}."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": "medium", "misconception_addressed": None},
            "memory": _make_memory(
                candidate=rng.random() > 0.6,
                title=f"Simple Definition: {concept}",
                content=f"{concept} simply means {info['explanation']}.",
                anchor_concept=concept,
                memory_type="definition",
                confidence=round(rng.uniform(0.7, 0.9), 2),
            ),
        },
    )


def create_analogy(domain, concept, info, seq, i, rng):
    turns = _get_student_turns("analogy", concept, info, rng)
    analogy = info["analogies"][i % len(info["analogies"])] if info.get("analogies") else f"{concept} is like an everyday process"
    return _make_record(
        "analogy", domain, concept, seq, info, rng,
        instruction=f"Explain {concept} using an analogy that connects to everyday experience.",
        conversation_turns=turns,
        target={
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "analogy",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"Think of it this way: {analogy}. This metaphor illuminates the fundamental nature of {concept}: {info['explanation']}. Connecting it to this analogy clarifies why {info['reason']}.",
            "understanding_check": {
                "required": True,
                "question": f"How does the analogy relate to {concept}?",
                "expected_answer": f"The analogy illustrates that {info['explanation']}."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": rng.choice(["low", "medium"]), "misconception_addressed": None},
            "memory": _make_memory(
                candidate=rng.random() > 0.6,
                title=f"Analogy for {concept}",
                content=f"Remember: {analogy}",
                anchor_concept=concept,
                memory_type="analogy",
                confidence=round(rng.uniform(0.7, 0.9), 2),
            ),
        },
    )


def create_concrete_example(domain, concept, info, seq, i, rng):
    turns = _get_student_turns("concrete_example", concept, info, rng)
    examples = info["concrete_examples"]
    ex1 = examples[i % len(examples)]
    ex2 = examples[(i + 1) % len(examples)]
    return _make_record(
        "concrete_example", domain, concept, seq, info, rng,
        instruction=f"Provide a concrete, real-world example of {concept}.",
        conversation_turns=turns,
        target={
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "concrete_example",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"To see how {concept} functions in reality, consider this case: {ex1}. In addition, {ex2}. These scenarios demonstrate in practice that {info['explanation']}.",
            "understanding_check": {
                "required": True,
                "question": f"Based on the examples, how would you describe {concept}?",
                "expected_answer": f"{concept} is {info['explanation']}."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": rng.choice(["low", "medium"]), "misconception_addressed": None},
            "memory": _make_memory(
                candidate=rng.random() > 0.5,
                title=f"Example of {concept}",
                content=f"Key example: {ex1}",
                anchor_concept=concept,
                memory_type="example",
                confidence=round(rng.uniform(0.6, 0.8), 2),
            ),
        },
    )


def create_numerical_example(domain, concept, info, seq, i, rng):
    turns = _get_student_turns("numerical_example", concept, info, rng)
    nums = info["numerical_examples"]
    n1 = nums[i % len(nums)]
    n2 = nums[(i + 1) % len(nums)]
    return _make_record(
        "numerical_example", domain, concept, seq, info, rng,
        instruction=f"Provide a worked numerical example demonstrating {concept}.",
        conversation_turns=turns,
        target={
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "concrete_example",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"Here is a concrete numerical walkthrough for {concept}: {n1}. Furthermore, consider: {n2}. Calculating through these values demonstrates precisely how {info['explanation']}.",
            "understanding_check": {
                "required": True,
                "question": f"Based on the numerical examples, what does {concept} demonstrate?",
                "expected_answer": f"{concept} demonstrates that {info['explanation']}."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": rng.choice(["low", "medium"]), "misconception_addressed": None},
            "memory": _make_memory(
                candidate=rng.random() > 0.5,
                title=f"Numerical example: {concept}",
                content=f"Key calculation: {n1}",
                anchor_concept=concept,
                memory_type="example",
                confidence=round(rng.uniform(0.6, 0.8), 2),
            ),
        },
    )


def create_diagnostic_questions(domain, concept, info, seq, i, rng):
    turns = _get_student_turns("diagnostic_questions", concept, info, rng)
    diag = info["diagnostic_questions"]
    q1 = diag[0] if diag else f"What part of {concept} confuses you most?"
    q2 = diag[1] if len(diag) > 1 else f"Can you explain what you think {concept} means?"
    return _make_record(
        "diagnostic_questions", domain, concept, seq, info, rng,
        instruction=f"Ask diagnostic questions to identify the nature and source of the student's confusion about {concept}.",
        conversation_turns=turns,
        target={
            "task": "resolve_confusion",
            "action": "diagnose",
            "strategy": "diagnostic_questions",
            "difficulty": "beginner",
            "response": f"Let us systematically diagnose your current understanding of {concept}: 1) {q1} 2) {q2} 3) How do you currently distinguish {concept} from related principles in {domain}? 4) Which specific step or definition feels least intuitive?",
            "understanding_check": {"required": False, "question": None, "expected_answer": None},
            "diagnosis": {"confusion_type": "conceptual", "severity": "medium", "misconception_addressed": None},
            "memory": _make_memory(candidate=False),
        },
    )


def create_follow_up_questions(domain, concept, info, seq, i, rng):
    turns = _get_student_turns("follow_up_questions", concept, info, rng)
    fqs = info["follow_up_questions"]
    q1 = fqs[0] if fqs else f"How would you apply {concept} in a real situation?"
    q2 = fqs[1] if len(fqs) > 1 else f"What are the limitations of {concept}?"
    return _make_record(
        "follow_up_questions", domain, concept, seq, info, rng,
        instruction=f"Generate follow-up questions to deepen the student's understanding of {concept}.",
        conversation_turns=turns,
        target={
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "follow_up_questions",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"Building on your foundational knowledge of {concept}, let us explore deeper implications: 1) {q1} 2) {q2} 3) Under what conditions might standard assumptions about {concept} need adjustment? Pondering these will strengthen your analytical mastery.",
            "understanding_check": {
                "required": True,
                "question": f"What is a good question to test deep understanding of {concept}?",
                "expected_answer": f"A good question would probe application and limitations, not just definition."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": "low", "misconception_addressed": None},
            "memory": _make_memory(candidate=False),
        },
    )


def create_prerequisite_detection(domain, concept, info, seq, i, rng):
    turns = _get_student_turns("prerequisite_detection", concept, info, rng)
    prereqs = info["prerequisites"]
    missing_prereq = prereqs[i % len(prereqs)] if prereqs else "foundational concepts"
    return _make_record(
        "prerequisite_detection", domain, concept, seq, info, rng,
        instruction=f"Identify what prerequisite knowledge the student needs to understand {concept}.",
        conversation_turns=turns,
        target={
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "prerequisite",
            "difficulty": "beginner",
            "response": f"Before mastering {concept}, it is critical to ensure a firm grasp of {missing_prereq}. Without understanding {missing_prereq}, {info['explanation']} can feel confusing. Let us review the foundational aspects of {missing_prereq} first.",
            "understanding_check": {
                "required": True,
                "question": f"What prerequisite is essential for understanding {concept}?",
                "expected_answer": f"{missing_prereq} is essential because {concept} builds on it."
            },
            "diagnosis": {
                "confusion_type": "prerequisite_gap",
                "severity": "medium",
                "misconception_addressed": None,
            },
            "memory": _make_memory(
                candidate=rng.random() > 0.6,
                title=f"Prerequisite for {concept}",
                content=f"You must understand {missing_prereq} before learning {concept}.",
                anchor_concept=concept,
                memory_type="prerequisite",
                confidence=round(rng.uniform(0.7, 0.9), 2),
            ),
        },
    )


def create_misconception_correction(domain, concept, info, seq, i, rng):
    turns = _get_student_turns("misconception_detection", concept, info, rng)
    misconceptions = info["misconceptions"]
    misc_pair = misconceptions[i % len(misconceptions)]
    misconception, refutation = misc_pair
    return _make_record(
        "misconception_detection", domain, concept, seq, info, rng,
        instruction=f"Address and correct the student's misconception about {concept}.",
        conversation_turns=turns,
        target={
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "misconception_correction",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"A frequent point of confusion is thinking that '{misconception}'. In reality, {refutation} The rigorous understanding of {concept} is that {info['explanation']}.",
            "understanding_check": {
                "required": True,
                "question": f"Why is the statement '{misconception}' incorrect?",
                "expected_answer": f"It's incorrect because {refutation}"
            },
            "diagnosis": {
                "confusion_type": "misconception",
                "severity": "medium",
                "misconception_addressed": misconception,
            },
            "memory": _make_memory(
                candidate=rng.random() > 0.6,
                title=f"Misconception correction: {concept}",
                content=f"Remember: '{misconception}' is WRONG. {refutation}",
                anchor_concept=concept,
                memory_type="misconception_correction",
                confidence=round(rng.uniform(0.7, 0.9), 2),
            ),
        },
    )


def create_multi_turn_tutoring(domain, concept, info, seq, i, rng):
    turns = _get_student_turns("multi_turn_tutoring", concept, info, rng)
    ex = info["concrete_examples"][i % len(info["concrete_examples"])]
    return _make_record(
        "multi_turn_tutoring", domain, concept, seq, info, rng,
        instruction=f"Continue the multi-turn tutoring conversation about {concept}.",
        conversation_turns=turns,
        target={
            "action": "explain",
            "strategy": "multi_turn_tutoring",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"That touches on a central principle of {concept}. In practice, {info['reason']}. Specifically: {info['explanation']}. To observe this dynamic at work: {ex}",
            "understanding_check": {
                "required": True,
                "question": f"Based on our conversation, how would you explain {concept}?",
                "expected_answer": f"{concept} means {info['explanation']}."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": "low", "misconception_addressed": None},
            "memory": _make_memory(
                candidate=rng.random() > 0.5,
                title=f"Multi-turn insight: {concept}",
                content=f"From our discussion: {info['explanation']}.",
                anchor_concept=concept,
                memory_type="insight",
                confidence=round(rng.uniform(0.6, 0.8), 2),
            ),
        },
    )


def create_repeated_confusion(domain, concept, info, seq, i, rng):
    turns = _get_student_turns("repeated_confusion", concept, info, rng)
    analogy = info["analogies"][0] if info.get("analogies") else f"a simple everyday process"
    ex = info["concrete_examples"][i % len(info["concrete_examples"])]
    return _make_record(
        "repeated_confusion", domain, concept, seq, info, rng,
        instruction=f"Address repeated confusion about {concept} using a completely different approach.",
        conversation_turns=turns,
        target={
            "action": "explain",
            "strategy": "simple_explanation",
            "difficulty": "beginner",
            "response": f"I recognize that {concept} can be challenging to absorb on the first few passes. Let us approach it from a fresh angle: {analogy}. At its heart, {concept} centers on {info['explanation']}. Consider this practical scenario: {ex}. Focus on how the pieces connect here rather than abstract definitions.",
            "understanding_check": {
                "required": True,
                "question": f"What was the key insight from our new approach to {concept}?",
                "expected_answer": f"The key was seeing {concept} through a practical example rather than abstract definitions."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": "medium", "misconception_addressed": None},
            "memory": _make_memory(
                candidate=rng.random() > 0.6,
                title=f"Breakthrough: {concept}",
                content=f"When confused about {concept}, remember: {ex}",
                anchor_concept=concept,
                memory_type="strategy",
                confidence=round(rng.uniform(0.7, 0.9), 2),
            ),
        },
    )


def create_strategy_switching(domain, concept, info, seq, i, rng):
    turns = _get_student_turns("strategy_switching", concept, info, rng)
    analogy = info["analogies"][0] if info.get("analogies") else f"Think of it like an everyday process"
    ex = info["concrete_examples"][i % len(info["concrete_examples"])]
    return _make_record(
        "strategy_switching", domain, concept, seq, info, rng,
        instruction=f"Switch teaching strategies to better explain {concept} after the initial approach failed.",
        conversation_turns=turns,
        target={
            "action": "explain",
            "strategy": "analogy",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"Let us shift our instructional strategy to clarify {concept}. {analogy}. This perspective helps unpack {concept} because {info['explanation']}. Additionally, notice this practical case: {ex}. Does visualizing it through this comparison make the relationship clearer?",
            "understanding_check": {
                "required": True,
                "question": f"Why did switching to an analogy help explain {concept}?",
                "expected_answer": "The analogy made the abstract concept more concrete and relatable."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": "medium", "misconception_addressed": None},
            "memory": _make_memory(
                candidate=rng.random() > 0.6,
                title=f"Strategy switch for {concept}",
                content=f"When standard explanations don't work for {concept}, try: {analogy}",
                anchor_concept=concept,
                memory_type="strategy",
                confidence=round(rng.uniform(0.7, 0.9), 2),
            ),
        },
    )


def create_hint_based_teaching(domain, concept, info, seq, i, rng):
    turns = _get_student_turns("hint_based_teaching", concept, info, rng)
    hints = info.get("hints", [])
    h1 = hints[0] if hints else f"Think about what {concept} really means."
    h2 = hints[1] if len(hints) > 1 else f"Consider how {concept} relates to everyday experience."
    ex = info["concrete_examples"][i % len(info["concrete_examples"])]
    return _make_record(
        "hint_based_teaching", domain, concept, seq, info, rng,
        instruction=f"Provide hints to help the learner figure out {concept} themselves.",
        conversation_turns=turns,
        target={
            "action": "explain",
            "strategy": "hint",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"Rather than walking through the entire answer, let me offer clues to help you deduce {concept} yourself. Clue 1: {h1}. Clue 2: {h2}. Clue 3: Consider this scenario: {ex}. Reflect on these clues and try describing {concept} in your own words.",
            "understanding_check": {
                "required": True,
                "question": f"Based on the hints, what is {concept}?",
                "expected_answer": f"{info['explanation']}."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": "medium", "misconception_addressed": None},
            "memory": _make_memory(
                candidate=rng.random() > 0.5,
                title=f"Hints for {concept}",
                content=f"Key hints: {h1} and {h2}.",
                anchor_concept=concept,
                memory_type="hint",
                confidence=round(rng.uniform(0.6, 0.8), 2),
            ),
        },
    )


def create_practice_questions(domain, concept, info, seq, i, rng):
    turns = _get_student_turns("practice_questions", concept, info, rng)
    problems = info.get("practice_problems", [])
    p1 = problems[0] if problems else f"Apply {concept} to solve this problem."
    p2 = problems[1] if len(problems) > 1 else f"Explain {concept} in your own words."
    return _make_record(
        "practice_questions", domain, concept, seq, info, rng,
        instruction=f"Provide practice problems to reinforce understanding of {concept}.",
        conversation_turns=turns,
        target={
            "action": "explain",
            "strategy": "practice",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"Let us reinforce your grasp of {concept} through targeted practice problems:\n\nExercise 1: {p1}\n\nExercise 2: {p2}\n\nExercise 3: Explain how {concept} would apply in a novel {domain} context.\n\nWork through each exercise and we will review your logic together.",
            "understanding_check": {
                "required": True,
                "question": f"What key skill did these problems test about {concept}?",
                "expected_answer": f"The key skill is applying {concept}: {info['explanation']}."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": "low", "misconception_addressed": None},
            "memory": _make_memory(
                candidate=rng.random() > 0.5,
                title=f"Practice for {concept}",
                content=f"Key practice types: calculation, explanation, application.",
                anchor_concept=concept,
                memory_type="procedure",
                confidence=round(rng.uniform(0.6, 0.8), 2),
            ),
        },
    )


def create_error_correction(domain, concept, info, seq, i, rng):
    turns = _get_student_turns("error_correction", concept, info, rng)
    misc_pair = info["misconceptions"][0]
    misconception, refutation = misc_pair
    return _make_record(
        "error_correction", domain, concept, seq, info, rng,
        instruction=f"Correct an error in the student's understanding of {concept}.",
        conversation_turns=turns,
        target={
            "action": "explain",
            "strategy": "misconception_correction",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"Looking closely at your reasoning, there is a key misconception to address. The idea that '{misconception}' is inaccurate. {refutation} The correct principle for {concept} is that {info['explanation']}.",
            "understanding_check": {
                "required": True,
                "question": f"What was the error in thinking about {concept}?",
                "expected_answer": f"The error was assuming '{misconception}', but {refutation}"
            },
            "diagnosis": {
                "confusion_type": "misconception",
                "severity": "medium",
                "misconception_addressed": misconception,
            },
            "memory": _make_memory(
                candidate=rng.random() > 0.6,
                title=f"Corrected understanding: {concept}",
                content=f"Remember: {info['explanation']}, NOT '{misconception}'.",
                anchor_concept=concept,
                memory_type="misconception_correction",
                confidence=round(rng.uniform(0.7, 0.9), 2),
            ),
        },
    )


def create_partial_understanding(domain, concept, info, seq, i, rng):
    turns = _get_student_turns("partial_understanding", concept, info, rng)
    return _make_record(
        "partial_understanding", domain, concept, seq, info, rng,
        instruction=f"Address the student's partial understanding of {concept} and fill in the gaps.",
        conversation_turns=turns,
        target={
            "action": "explain",
            "strategy": "partial_understanding",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"You have grasped an important component of {concept}—specifically, that {info['explanation']}. To complete the picture, observe that {info['reason']}. True mastery of {concept} brings together both the core mechanism ({info['explanation']}) and its broader impact ({info['reason']}).",
            "understanding_check": {
                "required": True,
                "question": f"What aspect of {concept} were you missing?",
                "expected_answer": f"The missing aspect was understanding that {info['reason']}."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": "low", "misconception_addressed": None},
            "memory": _make_memory(
                candidate=rng.random() > 0.5,
                title=f"Complete understanding: {concept}",
                content=f"To fully understand {concept}: {info['explanation']} AND {info['reason']}.",
                anchor_concept=concept,
                memory_type="concept",
                confidence=round(rng.uniform(0.6, 0.8), 2),
            ),
        },
    )


def create_understanding_confirmation(domain, concept, info, seq, i, rng):
    turns = _get_student_turns("understanding_confirmation", concept, info, rng)
    ex = info["concrete_examples"][i % len(info["concrete_examples"])]
    return _make_record(
        "understanding_confirmation", domain, concept, seq, info, rng,
        instruction=f"Confirm the student's understanding of {concept} and reinforce key points.",
        conversation_turns=turns,
        target={
            "action": "explain",
            "strategy": "understanding_confirmation",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"Well done! Your summary shows a precise and nuanced grasp of {concept}. You accurately identified that {info['explanation']}, and correctly recognized its importance: {info['reason']}. You are in a strong position to apply this concept to advanced problems.",
            "understanding_check": {
                "required": True,
                "question": f"How would you explain {concept} to someone else?",
                "expected_answer": f"I would explain that {info['explanation']} and that {info['reason']}."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": "low", "misconception_addressed": None},
            "memory": _make_memory(
                candidate=rng.random() > 0.7,
                title=f"Mastered: {concept}",
                content=f"You now understand {concept}: {info['explanation']}. Example: {ex}",
                anchor_concept=concept,
                memory_type="mastery",
                confidence=round(rng.uniform(0.8, 0.95), 2),
            ),
        },
    )


def create_memory_generation(domain, concept, info, seq, i, rng):
    turns = _get_student_turns("memory_generation", concept, info, rng)
    ex = info["concrete_examples"][i % len(info["concrete_examples"])]
    formulas = info.get("formulas", [])
    formula_text = f" Key formula(s): {'; '.join(formulas[:2])}." if formulas else ""
    return _make_record(
        "memory_generation", domain, concept, seq, info, rng,
        instruction=f"Generate a memorable summary for the student to review {concept} later.",
        conversation_turns=turns,
        target={
            "action": "explain",
            "strategy": "simple_explanation",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"Here is your structured summary card for {concept}:\n\n- Definition: {info['explanation']}.\n- Significance: {info['reason']}.\n- Illustrative Example: {ex}.{formula_text}\n- Key Reminder: {info['hints'][0] if info.get('hints') else 'Focus on the underlying mechanism rather than rote memorization.'}",
            "understanding_check": {
                "required": True,
                "question": f"What are the key points to remember about {concept}?",
                "expected_answer": f"{concept} means {info['explanation']} and matters because {info['reason']}."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": "low", "misconception_addressed": None},
            "memory": _make_memory(
                candidate=True,
                title=f"Essential: {concept}",
                content=f"{concept}: {info['explanation']}. Why: {info['reason']}. Example: {ex}",
                anchor_concept=concept,
                memory_type="definition",
                confidence=round(rng.uniform(0.8, 0.95), 2),
            ),
        },
    )


def create_document_grounded_core(domain, concept, info, seq, i, rng):
    turns = _get_student_turns("document_grounded", concept, info, rng)
    doc_evidence = info["document_evidence"]
    ev1 = doc_evidence[0] if doc_evidence else f"The text explains: '{info['explanation']}'"
    ev2 = doc_evidence[1] if len(doc_evidence) > 1 else f"The passage notes that {info['reason']}."
    return _make_record(
        "document_grounded", domain, concept, seq, info, rng,
        instruction=f"Teach {concept} using specific evidence from the document.",
        conversation_turns=turns,
        target={
            "action": "explain",
            "strategy": "document_grounded",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"Examining the authoritative text for {concept}: {ev1} Additionally, the text notes: {ev2} This confirms that {info['explanation']}. The cited evidence directly substantiates that {info['reason']}.",
            "understanding_check": {
                "required": True,
                "question": f"What evidence from the document supports the explanation of {concept}?",
                "expected_answer": f"The document states: '{ev1}', which shows that {info['explanation']}."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": "low", "misconception_addressed": None},
            "memory": _make_memory(
                candidate=rng.random() > 0.5,
                title=f"Document insight: {concept}",
                content=f"From reading: {ev1}",
                anchor_concept=concept,
                memory_type="document_insight",
                confidence=round(rng.uniform(0.6, 0.8), 2),
            ),
        },
    )


def create_context_retention(domain, concept, info, seq, i, rng):
    other_concepts = [c for c in get_domain_concepts(domain) if c != concept]
    other_concept = rng.choice(other_concepts) if other_concepts else concept
    other_info = get_concept_info(domain, other_concept) if other_concept != concept else info

    turns = [
        {"role": "student", "content": f"Earlier we talked about {other_concept}. Now I'm confused about {concept}."},
        {"role": "tutor", "content": f"Let's connect {other_concept} to {concept}."},
    ]

    return _make_record(
        "context_retention", domain, concept, seq, info, rng,
        instruction=f"Demonstrate retention of earlier context about {other_concept} while explaining {concept}.",
        conversation_turns=turns,
        target={
            "action": "explain",
            "strategy": "simple_explanation",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"Excellent connection between {other_concept} and {concept}! Earlier we established that {other_info['explanation']}. This connects directly to {concept} because {info['explanation']}. Both principles share the common foundation that {info['reason']}.",
            "understanding_check": {
                "required": True,
                "question": f"How does {other_concept} relate to {concept}?",
                "expected_answer": f"They are connected because {info['reason']}."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": "low", "misconception_addressed": None},
            "memory": _make_memory(
                candidate=rng.random() > 0.5,
                title=f"Connection: {concept} and {other_concept}",
                content=f"{concept} and {other_concept} are related through {info['reason']}.",
                anchor_concept=concept,
                memory_type="connection",
                confidence=round(rng.uniform(0.6, 0.8), 2),
            ),
        },
    )


# ============================================================
# TARGETED GENERATORS (TRACKS 20 TO 24)
# ============================================================

def create_prior_error_correction(domain, concept, info, seq, i, rng):
    """
    Capability 1: Prior Error Correction.
    Category: "error_correction" (seq 005..008)
    Detects tutor's prior error, acknowledges mistake, provides factually accurate correction,
    explains correct concepts, and verifies student understanding.
    """
    scenario = get_prior_error_scenario(domain, concept, i)
    prior_claim = scenario["prior_erroneous_claim"]
    error_reason = scenario["error_reason"]
    correct_claim = scenario["correct_claim"]
    probe = scenario["understanding_probe"]
    exp_ans = scenario["expected_answer"]

    conversation_turns = [
        {"role": "student", "content": f"Can you help me understand {concept}?"},
        {"role": "tutor", "content": prior_claim},
        {"role": "student", "content": f"Wait, is that completely accurate for {concept}? That doesn't seem to match what I read."},
    ]

    response_text = (
        f"You are right to catch that, and I apologize for the confusion in my previous response. "
        f"I previously stated that {prior_claim}. However, that was an error because {error_reason}. "
        f"The correct fact is that {correct_claim}. "
        f"To verify our understanding: {probe}"
    )

    doc_context = {
        "document_id": f"{domain}_{concept.lower().replace(' ', '_').replace('(', '').replace(')', '')[:20]}_cor",
        "title": f"Corrective Principles: {concept}",
        "page": rng.randint(15, 180),
        "section": concept.split("(")[0].strip() if "(" in concept else concept,
        "selected_text": info["document_evidence"][0] if info.get("document_evidence") else f"Proper formulation of {concept}.",
        "surrounding_context": info["document_evidence"][1] if len(info.get("document_evidence", [])) > 1 else f"Core theory of {concept}.",
        "retrieved_evidence": [f"Standard rule: {info['explanation']}"],
    }

    return _make_record(
        category="error_correction",
        domain=domain,
        concept=concept,
        seq=seq,
        info=info,
        rng=rng,
        instruction=f"Acknowledge the prior tutor error regarding {concept}, provide the factually correct explanation, and verify student understanding.",
        conversation_turns=conversation_turns,
        target={
            "task": "error_correction",
            "action": "explain",
            "strategy": "error_correction",
            "difficulty": "intermediate",
            "response": response_text,
            "understanding_check": {
                "required": True,
                "question": probe,
                "expected_answer": exp_ans,
            },
            "diagnosis": {
                "confusion_type": "prior_tutor_error",
                "severity": "high",
                "misconception_addressed": error_reason,
            },
            "memory": _make_memory(
                candidate=True,
                title=f"Prior Error Correction: {concept}",
                content=f"Corrected prior inaccuracy: {error_reason}. Fact: {info['explanation']}.",
                anchor_concept=concept,
                memory_type="error_correction",
                confidence=0.95,
            ),
        },
        task_type="error_correction",
        level="intermediate",
        custom_doc_context=doc_context,
    )


def create_dg_direct_and_conflict(domain, concept, info, seq, i, rng):
    """
    Targeted Document Grounding Track 21: Direct Evidence & Conflict Handling.
    Category: "document_grounded" (seq 005..008)
    - If i % 2 == 0: Behavior A (Direct evidence use)
    - If i % 2 == 1: Behavior D (Conflict handling)
    """
    if i % 2 == 0:
        # Behavior A: Direct Evidence Use
        gp = get_grounded_paraphrase(domain, concept)
        doc_context = {
            "document_id": f"{domain}_{concept.lower().replace(' ', '_').replace('(', '').replace(')', '')[:20]}_dir",
            "title": gp["document_title"],
            "page": gp["page"],
            "section": gp["section"],
            "selected_text": gp["selected_text"],
            "surrounding_context": gp["surrounding_context"],
            "retrieved_evidence": gp["retrieved_evidence"],
        }
        turns = [
            {"role": "student", "content": f"What does the document specifically state about {concept}?"}
        ]
        retrieved_first = gp["retrieved_evidence"][0] if gp["retrieved_evidence"] else f"Core principle of {concept}"
        return _make_record(
            category="document_grounded",
            domain=domain,
            concept=concept,
            seq=seq,
            info=info,
            rng=rng,
            instruction=f"Answer the student's question about {concept} strictly using the provided document evidence.",
            conversation_turns=turns,
            target={
                "task": "resolve_confusion",
                "action": "explain",
                "strategy": "document_grounded",
                "difficulty": "intermediate",
                "response": f"According to {gp['document_title']} (Section: {gp['section']}, page {gp['page']}), '{gp['selected_text']}' Furthermore, '{retrieved_first}'. This directly demonstrates that {info['explanation']}.",
                "understanding_check": {
                    "required": True,
                    "question": f"What specific evidence from the document defines {concept}?",
                    "expected_answer": f"The document excerpt states: '{gp['selected_text']}'."
                },
                "diagnosis": {"confusion_type": "conceptual", "severity": "low", "misconception_addressed": None},
                "memory": _make_memory(
                    candidate=True,
                    title=f"Evidence: {concept}",
                    content=f"Document evidence: {gp['selected_text']}",
                    anchor_concept=concept,
                    memory_type="document_insight",
                    confidence=0.88,
                ),
            },
            task_type="resolve_confusion",
            level="intermediate",
            custom_doc_context=doc_context,
        )
    else:
        # Behavior D: Conflict Handling
        ch = get_conflict_handling(domain, concept)
        conflicting_ev = ch.get("conflicting_evidence") or ch.get("retrieved_evidence", [f"Alternative perspective on {concept}"])
        doc_context = {
            "document_id": f"{domain}_{concept.lower().replace(' ', '_').replace('(', '').replace(')', '')[:20]}_cnf",
            "title": ch["document_title"],
            "page": ch["page"],
            "section": ch["section"],
            "selected_text": ch["selected_text"],
            "surrounding_context": ch["surrounding_context"],
            "retrieved_evidence": conflicting_ev,
        }
        turns = [
            {"role": "student", "content": ch["student_prompt"]}
        ]
        return _make_record(
            category="document_grounded",
            domain=domain,
            concept=concept,
            seq=seq,
            info=info,
            rng=rng,
            instruction=f"Resolve the apparent tension between document evidence sources regarding {concept}.",
            conversation_turns=turns,
            target={
                "task": "resolve_confusion",
                "action": "explain",
                "strategy": "document_grounded",
                "difficulty": "intermediate",
                "response": ch["resolution_explanation"],
                "understanding_check": {
                    "required": True,
                    "question": f"How is the apparent conflict between the sources resolved for {concept}?",
                    "expected_answer": f"The two sources apply to different contexts, accounting bases, or scopes as explained in the evidence."
                },
                "diagnosis": {"confusion_type": "conflicting_sources", "severity": "medium", "misconception_addressed": None},
                "memory": _make_memory(
                    candidate=True,
                    title=f"Source Reconciliation: {concept}",
                    content=f"Reconciled tension: {ch['resolution_explanation'][:120]}...",
                    anchor_concept=concept,
                    memory_type="resolution",
                    confidence=0.90,
                ),
            },
            task_type="resolve_confusion",
            level="intermediate",
            custom_doc_context=doc_context,
        )


def create_dg_grounded_paraphrase(domain, concept, info, seq, i, rng):
    """
    Targeted Document Grounding Track 22: Grounded Paraphrase (Behavior B).
    Category: "document_grounded" (seq 009..012)
    Paraphrases document text clearly and faithfully without adding unsupported assumptions.
    """
    gp = get_grounded_paraphrase(domain, concept)
    doc_context = {
        "document_id": f"{domain}_{concept.lower().replace(' ', '_').replace('(', '').replace(')', '')[:20]}_par",
        "title": gp["document_title"],
        "page": gp["page"],
        "section": gp["section"],
        "selected_text": gp["selected_text"],
        "surrounding_context": gp["surrounding_context"],
        "retrieved_evidence": gp["retrieved_evidence"],
    }
    turns = [
        {"role": "student", "content": gp["student_prompt"]}
    ]
    return _make_record(
        category="document_grounded",
        domain=domain,
        concept=concept,
        seq=seq,
        info=info,
        rng=rng,
        instruction=f"Paraphrase the document excerpt explaining {concept} clearly and accurately without adding unsupported claims.",
        conversation_turns=turns,
        target={
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "document_grounded",
            "difficulty": "beginner",
            "response": gp["faithful_paraphrase"],
            "understanding_check": {
                "required": True,
                "question": f"How does this paraphrase reflect the core meaning of the document text?",
                "expected_answer": f"It conveys that {info['explanation']} without introducing unsupported external claims."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": "low", "misconception_addressed": None},
            "memory": _make_memory(
                candidate=rng.random() > 0.5,
                title=f"Paraphrase: {concept}",
                content=f"Summary of document passage: {gp['faithful_paraphrase']}",
                anchor_concept=concept,
                memory_type="document_insight",
                confidence=0.85,
            ),
        },
        task_type="resolve_confusion",
        level="beginner",
        custom_doc_context=doc_context,
    )


def create_dg_evidence_priority(domain, concept, info, seq, i, rng):
    """
    Targeted Document Grounding Track 23: Evidence Prioritization (Behavior C).
    Category: "document_grounded" (seq 013..016)
    - If i in [0, 1]: Pattern A (Direct numerical / evidence extraction)
    - If i in [2, 3]: Pattern B (Mechanistic derivation)
    """
    pattern_key = "pattern_a_calc" if i in [0, 1] else "pattern_b_derive"
    ev_data = get_evidence_data(domain, concept, pattern_key)

    doc_context = {
        "document_id": f"{domain}_{concept.lower().replace(' ', '_').replace('(', '').replace(')', '')[:20]}_{pattern_key[:3]}",
        "title": ev_data["document_title"],
        "page": ev_data["page"],
        "section": ev_data["section"],
        "selected_text": ev_data["selected_text"],
        "surrounding_context": ev_data["surrounding_context"],
        "retrieved_evidence": ev_data["retrieved_evidence"],
    }
    turns = [
        {"role": "student", "content": ev_data["student_prompt"]}
    ]

    instruction_text = (
        f"Answer the question about {concept} by prioritizing the explicit factual values given in the retrieved evidence."
        if pattern_key == "pattern_a_calc"
        else f"Derive the answer step-by-step for {concept} using the formulas and inputs provided in the document evidence."
    )

    if pattern_key == "pattern_a_calc":
        probe_q = f"What specific result or threshold is established by the retrieved evidence for {concept}?"
        probe_ans = f"The evidence establishes: {ev_data.get('calculated_result', info['explanation'])}."
    else:
        probe_q = f"What formula or rule is applied step-by-step for {concept}?"
        probe_ans = f"The derivation applies: {ev_data.get('formula', info.get('formulas', ['core principle'])[0])}."

    return _make_record(
        category="document_grounded",
        domain=domain,
        concept=concept,
        seq=seq,
        info=info,
        rng=rng,
        instruction=instruction_text,
        conversation_turns=turns,
        target={
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "document_grounded",
            "difficulty": "intermediate",
            "response": ev_data["target_response"],
            "understanding_check": {
                "required": True,
                "question": probe_q,
                "expected_answer": probe_ans,
            },
            "diagnosis": {
                "confusion_type": "evidence_prioritization" if pattern_key == "pattern_a_calc" else "mechanistic_derivation",
                "severity": "medium",
                "misconception_addressed": None,
            },
            "memory": _make_memory(
                candidate=True,
                title=f"{'Direct Evidence Value' if pattern_key == 'pattern_a_calc' else 'Mechanistic Derivation'}: {concept}",
                content=f"Evidence application: {ev_data['target_response'][:120]}...",
                anchor_concept=concept,
                memory_type="evidence_calculation" if pattern_key == "pattern_a_calc" else "procedure",
                confidence=0.92,
            ),
        },
        task_type="resolve_confusion",
        level="intermediate",
        custom_doc_context=doc_context,
    )


def create_dg_absence_refusal(domain, concept, info, seq, i, rng):
    """
    Targeted Document Grounding Track 24: Document Absence / Refusal (Behavior E).
    Category: "document_grounded" (seq 017..020)
    Explicitly states requested fact is absent from document, refuses to invent,
    distinguishes document evidence from general knowledge, and identifies required sources.
    """
    ar = get_absence_refusal(domain, concept, i)

    doc_context = {
        "document_id": f"{domain}_{concept.lower().replace(' ', '_').replace('(', '').replace(')', '')[:20]}_abs",
        "title": ar["document_title"],
        "page": ar["page"],
        "section": ar["section"],
        "selected_text": ar["selected_text"],
        "surrounding_context": ar["surrounding_context"],
        "retrieved_evidence": ar["retrieved_evidence"],
    }
    turns = [
        {"role": "student", "content": ar["student_prompt"]}
    ]

    refusal_response = (
        f"Based strictly on the provided document ({ar['document_title']}, Section: {ar['section']}), "
        f"there is no mention or definition of '{ar['unmentioned_term']}'. "
        f"The provided text focuses on {ar['actual_topic']}. "
        f"To find verified information regarding '{ar['unmentioned_term']}', you would need to consult a specialized resource such as {ar['authoritative_source']}."
    )

    return _make_record(
        category="document_grounded",
        domain=domain,
        concept=concept,
        seq=seq,
        info=info,
        rng=rng,
        instruction=f"Evaluate whether the provided document evidence supports the student's question about {concept}, refusing to invent missing facts.",
        conversation_turns=turns,
        target={
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "document_grounded",
            "difficulty": "intermediate",
            "response": refusal_response,
            "understanding_check": {
                "required": True,
                "question": f"Does the provided document contain information about '{ar['unmentioned_term']}'?",
                "expected_answer": f"No, the provided document does not mention '{ar['unmentioned_term']}'; {ar['authoritative_source']} would be needed."
            },
            "diagnosis": {
                "confusion_type": "document_absence",
                "severity": "low",
                "misconception_addressed": f"Student assumed document addressed {ar['unmentioned_term']}",
            },
            "memory": _make_memory(candidate=False),
        },
        task_type="resolve_confusion",
        level="intermediate",
        custom_doc_context=doc_context,
    )


# ============================================================
# TRACK DEFINITION
# ============================================================

TRACK_GENERATORS = [
    # Tracks 0..19: Core canonical categories (seq 001..004)
    ("simple_explanation", create_simple_explanation, 1),
    ("simplification", create_simplification, 1),
    ("analogy", create_analogy, 1),
    ("concrete_example", create_concrete_example, 1),
    ("numerical_example", create_numerical_example, 1),
    ("diagnostic_questions", create_diagnostic_questions, 1),
    ("follow_up_questions", create_follow_up_questions, 1),
    ("prerequisite_detection", create_prerequisite_detection, 1),
    ("misconception_detection", create_misconception_correction, 1),
    ("multi_turn_tutoring", create_multi_turn_tutoring, 1),
    ("repeated_confusion", create_repeated_confusion, 1),
    ("strategy_switching", create_strategy_switching, 1),
    ("hint_based_teaching", create_hint_based_teaching, 1),
    ("practice_questions", create_practice_questions, 1),
    ("error_correction", create_error_correction, 1),
    ("partial_understanding", create_partial_understanding, 1),
    ("understanding_confirmation", create_understanding_confirmation, 1),
    ("memory_generation", create_memory_generation, 1),
    ("document_grounded", create_document_grounded_core, 1),
    ("context_retention", create_context_retention, 1),
    # Track 20: Prior Error Correction (seq 005..008)
    ("error_correction", create_prior_error_correction, 5),
    # Tracks 21..24: Targeted Document Grounding (seq 005..020)
    ("document_grounded", create_dg_direct_and_conflict, 5),
    ("document_grounded", create_dg_grounded_paraphrase, 9),
    ("document_grounded", create_dg_evidence_priority, 13),
    ("document_grounded", create_dg_absence_refusal, 17),
]

assert len(TRACK_GENERATORS) == 25, f"Expected 25 tracks, got {len(TRACK_GENERATORS)}"


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_v04_dataset() -> List[Dict[str, Any]]:
    """Generate exactly 600 records deterministically using seed 42."""
    rng = random.Random(SEED)
    all_records = []

    print("=" * 60)
    print("OLORIC v0.4 DATASET GENERATION")
    print("=" * 60)

    # Track concept counts
    concept_counter = Counter()

    for domain in VALID_DOMAINS:
        domain_concepts = get_domain_concepts(domain)
        assert len(domain_concepts) == 10, f"Domain {domain} must have exactly 10 concepts, got {len(domain_concepts)}"

        for track_index, (category, gen_func, base_seq) in enumerate(TRACK_GENERATORS):
            for i in range(4):
                # Deterministic concept rotation: concept_idx = (track_index * 4 + i) % 10
                concept_idx = (track_index * 4 + i) % len(domain_concepts)
                concept = domain_concepts[concept_idx]
                info = get_concept_info(domain, concept)

                seq = base_seq + i
                record = gen_func(domain, concept, info, seq, i, rng)

                # Integrity checks
                assert record["id"].startswith("v04_"), f"Invalid ID prefix in {record['id']}"
                assert record["category"] == category, f"Category mismatch: {record['category']} vs {category}"
                assert record["domain"] == domain, f"Domain mismatch: {record['domain']} vs {domain}"

                all_records.append(record)
                concept_counter[(domain, concept)] += 1

    print(f"Total records generated: {len(all_records)}")
    assert len(all_records) == TOTAL_RECORDS, f"Expected {TOTAL_RECORDS}, got {len(all_records)}"

    # Check ID uniqueness
    ids = [r["id"] for r in all_records]
    assert len(ids) == len(set(ids)), f"Duplicate IDs detected! {len(ids)} total vs {len(set(ids))} unique"

    # Verify per-concept counts (must be exactly 10 each)
    for (dom, conc), count in concept_counter.items():
        assert count == 10, f"Concept {dom}/{conc} has {count} records, expected 10"
    assert len(concept_counter) == 60, f"Expected 60 concepts, got {len(concept_counter)}"

    return all_records


def write_dataset(records: List[Dict[str, Any]], output_path: str):
    """Write records to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Successfully written {len(records)} records to {output_path}")


def create_splits(records: List[Dict[str, Any]], output_dir: str, seed: int = SEED) -> Dict[str, List[Dict[str, Any]]]:
    """Create deterministic train/validation/test splits (480 / 60 / 60)."""
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)

    indices = list(range(len(records)))
    rng.shuffle(indices)

    train_idx = indices[:TRAIN_COUNT]
    val_idx = indices[TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT]
    test_idx = indices[TRAIN_COUNT + VAL_COUNT:]

    splits = {
        "train.jsonl": [records[i] for i in train_idx],
        "validation.jsonl": [records[i] for i in val_idx],
        "test.jsonl": [records[i] for i in test_idx],
    }

    for fname, split_records in splits.items():
        path = os.path.join(output_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            for r in split_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {fname}: {len(split_records)} records")

    # Strict isolation check
    train_ids = {r["id"] for r in splits["train.jsonl"]}
    val_ids = {r["id"] for r in splits["validation.jsonl"]}
    test_ids = {r["id"] for r in splits["test.jsonl"]}

    assert len(train_ids) == TRAIN_COUNT, f"Expected {TRAIN_COUNT} train IDs"
    assert len(val_ids) == VAL_COUNT, f"Expected {VAL_COUNT} val IDs"
    assert len(test_ids) == TEST_COUNT, f"Expected {TEST_COUNT} test IDs"
    assert not (train_ids & val_ids), "Train and Validation sets overlap!"
    assert not (train_ids & test_ids), "Train and Test sets overlap!"
    assert not (val_ids & test_ids), "Validation and Test sets overlap!"

    print(f"Splits created: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    print("Split isolation check: PASS")
    return splits


def compute_sha256(filepath: str) -> Tuple[str, str]:
    """Compute both raw and LF-normalized SHA-256 hashes."""
    with open(filepath, "rb") as f:
        raw_bytes = f.read()
    raw_sha = hashlib.sha256(raw_bytes).hexdigest().upper()
    lf_bytes = raw_bytes.replace(b"\r\n", b"\n")
    lf_sha = hashlib.sha256(lf_bytes).hexdigest().upper()
    return raw_sha, lf_sha


def main():
    repo_root = Path(__file__).parent.parent
    dataset_path = repo_root / "data" / "generated" / "oloric_v04_dataset.jsonl"
    splits_dir = repo_root / "data" / "splits_v04"

    records = generate_v04_dataset()
    write_dataset(records, str(dataset_path))
    create_splits(records, str(splits_dir))

    raw_sha, lf_sha = compute_sha256(str(dataset_path))
    print(f"\nDataset SHA-256 (Raw): {raw_sha}")
    print(f"Dataset SHA-256 (LF-normalized): {lf_sha}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
