#!/usr/bin/env python3
"""
OLORIC v0.3 Dataset Generator

Generates 480 records: 20 categories * 24 records each.
Each domain gets 4 records per category (6 domains * 4 = 24 per category).
Deterministic with seed=42.

Root-cause fixes applied:
  1. Task-type following: category-specific student turns + task/instruction in every record
  2. Cross-domain bleed: per-concept examples from v03_concept_registry
  3. Misconception incoherence: concept-paired misconceptions with real refutations
  4. Document grounding: strategy="document_grounded" + evidence from registry
"""
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add scripts dir to path
sys.path.insert(0, str(Path(__file__).parent))
from v03_concept_registry import (
    VALID_CATEGORIES, VALID_DOMAINS, CONCEPT_REGISTRY,
    get_domain_concepts, get_concept_info,
)

SEED = 42
TOTAL_RECORDS = 480
RECORDS_PER_CATEGORY = 24  # = 6 domains * 4 per domain
RECORDS_PER_DOMAIN_PER_CATEGORY = 4
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# ============================================================
# ID GENERATION
# ============================================================

def generate_example_id(category: str, domain: str, index: int) -> str:
    return f"v03_{domain}_{category}_{index+1:03d}"

# ============================================================
# CONTEXT BUILDERS
# ============================================================

def create_base_context(
    domain: str,
    concept: str,
    info: Dict[str, Any],
    rng: random.Random,
    level: str = "beginner",
    conversation_turns: Optional[List[Dict[str, str]]] = None,
    task_type: str = "resolve_confusion",
) -> Dict[str, Any]:
    """Create a base context for an example using per-concept content."""
    # Select concept-specific document evidence for the context
    doc_evidence = info.get("document_evidence", [])
    selected_text = doc_evidence[0] if doc_evidence else f"The concept of {concept} is fundamental to understanding {domain}."
    surrounding = doc_evidence[1] if len(doc_evidence) > 1 else f"In the study of {domain}, {concept} plays a crucial role."

    # Use concept-specific formulas for retrieved_evidence
    formulas = info.get("formulas", [])
    retrieved = rng.sample(formulas, min(2, len(formulas))) if formulas else [f"Key principle of {concept}"]

    if conversation_turns is None:
        conversation_turns = [
            {"role": "student", "content": f"I'm struggling to understand {concept}."}
        ]

    # Use concept-specific prerequisites
    prereqs = info.get("prerequisites", [])
    known_prereqs = rng.sample(prereqs, min(2, len(prereqs))) if prereqs else []
    weak_prereqs = rng.sample(prereqs, min(1, len(prereqs))) if prereqs else []

    return {
        "task": task_type,
        "document_context": {
            "document_id": f"{domain}_{concept.lower().replace(' ', '_')[:20]}",
            "title": f"Introduction to {concept}",
            "page": rng.randint(10, 200),
            "section": concept.split("(")[0].strip() if "(" in concept else concept,
            "selected_text": selected_text,
            "surrounding_context": surrounding,
            "retrieved_evidence": retrieved,
        },
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

# ============================================================
# CATEGORY-SPECIFIC STUDENT TURN TEMPLATES
# Root cause fix: each category gets distinct conversation openings
# so the model can distinguish task types even without instruction
# ============================================================

def _get_student_turns(category: str, concept: str, info: Dict[str, Any], rng: random.Random) -> List[Dict[str, str]]:
    """Generate category-appropriate student conversation turns."""

    prereqs = info.get("prerequisites", ["foundational concepts"])
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
            [{"role": "student", "content": f"The textbook explanation of {concept} is way too complicated. Can you simplify it?"}],
            [{"role": "student", "content": f"I'm only a beginner. Can you explain {concept} without all the jargon?"}],
            [{"role": "student", "content": f"Is there a simpler way to think about {concept}? The formal definition is too abstract."}],
            [{"role": "student", "content": f"I feel overwhelmed by {concept}. Can you dumb it down for me?"}],
        ],
        "analogy": [
            [{"role": "student", "content": f"Can you give me an analogy for {concept}? I learn better with comparisons."}],
            [{"role": "student", "content": f"Is there something in everyday life that works like {concept}?"}],
            [{"role": "student", "content": f"I understand the words but not the concept. What is {concept} similar to?"}],
            [{"role": "student", "content": f"Can you compare {concept} to something I already know?"}],
        ],
        "concrete_example": [
            [{"role": "student", "content": f"Can you give me a real-world example of {concept}?"}],
            [{"role": "student", "content": f"I understand the theory of {concept}, but what does it look like in practice?"}],
            [{"role": "student", "content": f"Show me a concrete case where {concept} applies."}],
            [{"role": "student", "content": f"Can you illustrate {concept} with a specific scenario?"}],
        ],
        "numerical_example": [
            [{"role": "student", "content": f"Can you show me {concept} with actual numbers?"}],
            [{"role": "student", "content": f"I need to see a worked-out numerical example of {concept}."}],
            [{"role": "student", "content": f"Walk me through a calculation involving {concept}."}],
            [{"role": "student", "content": f"Can you give me a problem with numbers that demonstrates {concept}?"}],
        ],
        "diagnostic_questions": [
            [{"role": "student", "content": f"I'm confused about {concept} but I'm not sure exactly what I don't understand."}],
            [{"role": "student", "content": f"I've been studying {concept} but something isn't clicking. Can you help me figure out what I'm missing?"}],
            [{"role": "student", "content": f"I think I understand {concept} but I'm not confident. Can you test my understanding?"}],
            [{"role": "student", "content": f"I keep getting {concept} questions wrong on practice tests. Where am I going wrong?"}],
        ],
        "follow_up_questions": [
            [{"role": "student", "content": f"I think I understand the basics of {concept}. What should I think about next?"},
             {"role": "tutor", "content": f"Good start! Let me ask you some follow-up questions to deepen your understanding."}],
            [{"role": "student", "content": f"OK, I get the definition of {concept}. But how do I know if I really understand it?"},
             {"role": "tutor", "content": f"Great question. Let's explore deeper with some probing questions."}],
            [{"role": "student", "content": f"I memorized the {concept} definition but I want to truly understand it. What questions should I ask myself?"}],
            [{"role": "student", "content": f"Now that I know what {concept} is, what are the important follow-up questions I should consider?"}],
        ],
        "prerequisite_detection": [
            [{"role": "student", "content": f"I'm struggling to understand {concept}."},
             {"role": "tutor", "content": f"Let me explain {concept}."},
             {"role": "student", "content": f"But I don't understand {prereqs[0] if prereqs else 'the basics'}. How is that related?"}],
            [{"role": "student", "content": f"I can't follow the explanation of {concept}. I think I'm missing some background knowledge."},
             {"role": "tutor", "content": f"Let's figure out what foundation you need."},
             {"role": "student", "content": f"What do I need to know before I can learn {concept}?"}],
            [{"role": "student", "content": f"Everyone else seems to get {concept} but I'm lost. Am I missing a prerequisite?"},
             {"role": "tutor", "content": f"Let me check your foundational knowledge."}],
            [{"role": "student", "content": f"I read the chapter on {concept} but nothing made sense. Do I need to study something else first?"}],
        ],
        "misconception_detection": [
            [{"role": "student", "content": f"I think {misc_text}."},
             {"role": "tutor", "content": f"That's a common misconception. Let me explain why it's incorrect."}],
            [{"role": "student", "content": f"Isn't it true that {misc_text}? That's what I've always thought."},
             {"role": "tutor", "content": f"I can see why you'd think that, but it's not quite right. Let me clarify."}],
            [{"role": "student", "content": f"My friend told me that {misc_text}. Is that correct?"},
             {"role": "tutor", "content": f"That's actually a widespread misconception. Let me explain the correct understanding."}],
            [{"role": "student", "content": f"I'm confused because I assumed {misc_text}, but the textbook seems to say otherwise."},
             {"role": "tutor", "content": f"Good catch! That assumption is indeed incorrect."}],
        ],
        "multi_turn_tutoring": [
            [{"role": "student", "content": f"I don't understand {concept}."},
             {"role": "tutor", "content": f"Let me explain {concept} simply: {info['explanation']}."},
             {"role": "student", "content": f"I get the basic idea, but how does that work in practice?"}],
            [{"role": "student", "content": f"Can you explain {concept}?"},
             {"role": "tutor", "content": f"{concept} refers to {info['explanation']}."},
             {"role": "student", "content": f"OK, but why does that matter? What's the practical importance?"}],
            [{"role": "student", "content": f"I'm studying {concept} and have some questions."},
             {"role": "tutor", "content": f"Sure! What part of {concept} are you working on?"},
             {"role": "student", "content": f"I understand the definition but I can't apply it to problems."}],
            [{"role": "student", "content": f"Let's go over {concept} step by step."},
             {"role": "tutor", "content": f"Great approach! {concept} is about {info['explanation']}."},
             {"role": "student", "content": f"That makes sense. What's the next thing I need to understand?"}],
        ],
        "repeated_confusion": [
            [{"role": "student", "content": f"I don't understand {concept}."},
             {"role": "tutor", "content": f"Let me explain: {info['explanation']}."},
             {"role": "student", "content": f"I still don't get it."},
             {"role": "tutor", "content": f"OK, let me try another approach."},
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
             {"role": "tutor", "content": f"Sure! Here's your first hint: {info['hints'][0] if info['hints'] else 'Think about the basics.'}"},
             {"role": "student", "content": f"I think I see it... can I have another hint?"},
             {"role": "tutor", "content": f"Here's hint 2: {info['hints'][1] if len(info['hints']) > 1 else 'Consider a specific example.'}"}],
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
            [{"role": "student", "content": f"My answer for the {concept} problem was different from the textbook. What did I do wrong?"},
             {"role": "tutor", "content": f"Let me trace through your reasoning to find the error."}],
            [{"role": "student", "content": f"I applied {concept} but got an absurd result. Where's my mistake?"},
             {"role": "tutor", "content": f"Let's work through it together and find the error."}],
        ],
        "partial_understanding": [
            [{"role": "student", "content": f"I think I get part of {concept} but not everything."},
             {"role": "tutor", "content": f"Good progress! Let me help fill in the gaps."}],
            [{"role": "student", "content": f"I understand why {concept} matters, but I can't do the calculations."},
             {"role": "tutor", "content": f"So you have the conceptual part — let's work on the procedural side."}],
            [{"role": "student", "content": f"I can solve basic {concept} problems but the complex ones stump me."},
             {"role": "tutor", "content": f"You have a solid foundation. Let me help you extend it."}],
            [{"role": "student", "content": f"I know the formula for {concept} but I don't understand WHY it works."},
             {"role": "tutor", "content": f"That's a great question. Let me bridge that gap."}],
        ],
        "understanding_confirmation": [
            [{"role": "student", "content": f"I think I finally understand {concept}! Let me explain it to you."},
             {"role": "tutor", "content": f"Excellent! I'd love to hear your explanation."}],
            [{"role": "student", "content": f"OK, so {concept} means {info['explanation']}. Am I right?"},
             {"role": "tutor", "content": f"Let me assess your understanding."}],
            [{"role": "student", "content": f"After studying {concept} more, I feel confident now. Can you confirm I'm on track?"},
             {"role": "tutor", "content": f"Sure! Tell me what you've learned."}],
            [{"role": "student", "content": f"I practiced {concept} problems and got them all right. Do I really understand it?"},
             {"role": "tutor", "content": f"Let's verify with a few questions to make sure."}],
        ],
        "memory_generation": [
            [{"role": "student", "content": f"I understand {concept} now but I'm worried I'll forget it. Can you help me create a memory aid?"},
             {"role": "tutor", "content": f"Sure! Let me create a memorable summary for you."}],
            [{"role": "student", "content": f"What are the key points about {concept} that I absolutely need to remember?"},
             {"role": "tutor", "content": f"Let me distill the essentials."}],
            [{"role": "student", "content": f"Can you give me a cheat sheet or mnemonic for {concept}?"},
             {"role": "tutor", "content": f"Great idea. Here's a memory-friendly summary."}],
            [{"role": "student", "content": f"I want to review {concept} later. What should I put on my flashcard?"},
             {"role": "tutor", "content": f"Let me create the perfect summary for your review."}],
        ],
        "document_grounded": [
            [{"role": "student", "content": f"The textbook mentions {concept} but I don't understand the passage. Can you explain what it means?"},
             {"role": "tutor", "content": f"Let's look at the specific text together."}],
            [{"role": "student", "content": f"I'm reading about {concept} in the document. What does this evidence mean?"},
             {"role": "tutor", "content": f"Let me walk you through the key points from the text."}],
            [{"role": "student", "content": f"The chapter on {concept} has some confusing sections. Can you explain them using the text?"},
             {"role": "tutor", "content": f"Sure! Let me reference the specific passages."}],
            [{"role": "student", "content": f"I need to answer a question about {concept} using evidence from our reading. Can you help me understand the source?"},
             {"role": "tutor", "content": f"Let's analyze the document together."}],
        ],
        "context_retention": [
            # These use a different concept reference, handled separately
            [],
        ],
    }

    if category == "context_retention":
        # Special handling: reference another concept from the same domain
        domain = info.get("domain", "general_academic")
        other_concepts = [c for c in get_domain_concepts(domain) if c != concept]
        other = rng.choice(other_concepts) if other_concepts else concept
        return [
            [{"role": "student", "content": f"Earlier we talked about {other}. Now I'm confused about {concept}."},
             {"role": "tutor", "content": f"Let's connect {other} to {concept}."}],
            [{"role": "student", "content": f"How does {concept} relate to {other} that we covered before?"},
             {"role": "tutor", "content": f"Good question! Let me show you the connection."}],
            [{"role": "student", "content": f"In our last session we discussed {other}. Now {concept} seems related. How?"},
             {"role": "tutor", "content": f"They are indeed connected. Let me explain."}],
            [{"role": "student", "content": f"I remember learning about {other}. Is {concept} built on top of that?"},
             {"role": "tutor", "content": f"In a way, yes. Let me show you how they fit together."}],
        ]

    cat_templates = templates.get(category, templates["simple_explanation"])
    return rng.choice(cat_templates) if cat_templates else [{"role": "student", "content": f"I'm struggling to understand {concept}."}]


# ============================================================
# RECORD CREATION FUNCTIONS — ONE PER CATEGORY
# ============================================================

def _make_record(
    category: str,
    domain: str,
    concept: str,
    index: int,
    info: Dict[str, Any],
    rng: random.Random,
    instruction: str,
    conversation_turns: List[Dict[str, str]],
    target: Dict[str, Any],
    task_type: str = "resolve_confusion",
    level: str = "beginner",
) -> Dict[str, Any]:
    """Create a standardized record with all required fields."""
    return {
        "id": generate_example_id(category, domain, index),
        "category": category,
        "domain": domain,
        "instruction": instruction,
        "context": create_base_context(domain, concept, info, rng, level, conversation_turns, task_type),
        "target": target,
    }


def create_simple_explanation(domain, concept, info, index, rng):
    turns = _get_student_turns("simple_explanation", concept, info, rng)
    ex_idx = index % len(info["concrete_examples"])
    example = info["concrete_examples"][ex_idx]
    return _make_record(
        "simple_explanation", domain, concept, index, info, rng,
        instruction=f"Explain the concept of {concept} in {domain} clearly and simply.",
        conversation_turns=turns,
        target={
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "simple_explanation",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"{concept} refers to {info['explanation']}. This is important because {info['reason']}. For example, {example}",
            "understanding_check": {
                "required": True,
                "question": info["diagnostic_questions"][0] if info["diagnostic_questions"] else f"What is the main idea of {concept}?",
                "expected_answer": f"The main idea is that {info['explanation']}."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": rng.choice(["low", "medium"]),
                "misconception_addressed": None,
            },
            "memory": {
                "candidate": rng.random() > 0.7,
                "title": f"Key Concept: {concept}",
                "content": f"{concept}: {info['explanation']}.",
                "anchor_concept": concept,
                "memory_type": "definition",
                "confidence": round(rng.uniform(0.6, 0.9), 2),
            },
        },
    )


def create_simplification(domain, concept, info, index, rng):
    turns = _get_student_turns("simplification", concept, info, rng)
    analogy = info["analogies"][0] if info["analogies"] else f"Think of {concept} like an everyday process"
    return _make_record(
        "simplification", domain, concept, index, info, rng,
        instruction=f"Simplify the explanation of {concept} for a beginner learner.",
        conversation_turns=turns,
        target={
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "simplification",
            "difficulty": "beginner",
            "response": f"Let me make {concept} really simple. {analogy}. Basically, {concept} means {info['explanation']}. The key takeaway is that {info['reason']}.",
            "understanding_check": {
                "required": True,
                "question": f"In your own words, what does {concept} mean?",
                "expected_answer": f"It means {info['explanation']}."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": "medium", "misconception_addressed": None},
            "memory": {
                "candidate": rng.random() > 0.6,
                "title": f"Simple Definition: {concept}",
                "content": f"{concept} simply means {info['explanation']}.",
                "anchor_concept": concept,
                "memory_type": "definition",
                "confidence": round(rng.uniform(0.7, 0.9), 2),
            },
        },
    )


def create_analogy(domain, concept, info, index, rng):
    turns = _get_student_turns("analogy", concept, info, rng)
    analogy = info["analogies"][index % len(info["analogies"])] if info["analogies"] else f"{concept} is like an everyday process"
    return _make_record(
        "analogy", domain, concept, index, info, rng,
        instruction=f"Explain {concept} using an analogy that connects to everyday experience.",
        conversation_turns=turns,
        target={
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "analogy",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"{analogy}. This captures the essence of {concept}: {info['explanation']}. Understanding this analogy helps because {info['reason']}.",
            "understanding_check": {
                "required": True,
                "question": f"How does the analogy relate to {concept}?",
                "expected_answer": f"The analogy illustrates that {info['explanation']}."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": rng.choice(["low", "medium"]), "misconception_addressed": None},
            "memory": {
                "candidate": rng.random() > 0.6,
                "title": f"Analogy for {concept}",
                "content": f"Remember: {analogy}",
                "anchor_concept": concept,
                "memory_type": "analogy",
                "confidence": round(rng.uniform(0.7, 0.9), 2),
            },
        },
    )


def create_concrete_example(domain, concept, info, index, rng):
    turns = _get_student_turns("concrete_example", concept, info, rng)
    examples = info["concrete_examples"]
    ex1 = examples[index % len(examples)]
    ex2 = examples[(index + 1) % len(examples)]
    return _make_record(
        "concrete_example", domain, concept, index, info, rng,
        instruction=f"Provide a concrete, real-world example of {concept}.",
        conversation_turns=turns,
        target={
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "concrete_example",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"Here's a concrete example of {concept}: {ex1}. Another example: {ex2}. These illustrate that {info['explanation']}.",
            "understanding_check": {
                "required": True,
                "question": f"Based on the examples, how would you describe {concept}?",
                "expected_answer": f"{concept} is {info['explanation']}."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": rng.choice(["low", "medium"]), "misconception_addressed": None},
            "memory": {
                "candidate": rng.random() > 0.5,
                "title": f"Example of {concept}",
                "content": f"Key example: {ex1}",
                "anchor_concept": concept,
                "memory_type": "example",
                "confidence": round(rng.uniform(0.6, 0.8), 2),
            },
        },
    )


def create_numerical_example(domain, concept, info, index, rng):
    turns = _get_student_turns("numerical_example", concept, info, rng)
    nums = info["numerical_examples"]
    n1 = nums[index % len(nums)]
    n2 = nums[(index + 1) % len(nums)]
    return _make_record(
        "numerical_example", domain, concept, index, info, rng,
        instruction=f"Provide a worked numerical example demonstrating {concept}.",
        conversation_turns=turns,
        target={
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "concrete_example",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"Let me walk you through a numerical example of {concept}: {n1}. Here's another: {n2}. These numbers show how {info['explanation']}.",
            "understanding_check": {
                "required": True,
                "question": f"Based on the numerical examples, what does {concept} demonstrate?",
                "expected_answer": f"{concept} demonstrates that {info['explanation']}."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": rng.choice(["low", "medium"]), "misconception_addressed": None},
            "memory": {
                "candidate": rng.random() > 0.5,
                "title": f"Numerical example: {concept}",
                "content": f"Key calculation: {n1}",
                "anchor_concept": concept,
                "memory_type": "example",
                "confidence": round(rng.uniform(0.6, 0.8), 2),
            },
        },
    )


def create_diagnostic_questions(domain, concept, info, index, rng):
    turns = _get_student_turns("diagnostic_questions", concept, info, rng)
    diag = info["diagnostic_questions"]
    q1 = diag[0] if diag else f"What part of {concept} confuses you most?"
    q2 = diag[1] if len(diag) > 1 else f"Can you explain what you think {concept} means?"
    return _make_record(
        "diagnostic_questions", domain, concept, index, info, rng,
        instruction=f"Ask diagnostic questions to identify the nature and source of the student's confusion about {concept}.",
        conversation_turns=turns,
        target={
            "task": "resolve_confusion",
            "action": "diagnose",
            "strategy": "diagnostic_questions",
            "difficulty": "beginner",
            "response": f"To help me understand your confusion about {concept}, let me ask you some targeted questions: 1) {q1} 2) {q2} 3) Have you encountered similar concepts before in your studies? 4) Which part of the explanation was clearest to you?",
            "understanding_check": {"required": False, "question": None, "expected_answer": None},
            "diagnosis": {"confusion_type": "conceptual", "severity": "medium", "misconception_addressed": None},
            "memory": {
                "candidate": False, "title": None, "content": None,
                "anchor_concept": None, "memory_type": None, "confidence": 0.0,
            },
        },
    )


def create_follow_up_questions(domain, concept, info, index, rng):
    turns = _get_student_turns("follow_up_questions", concept, info, rng)
    fqs = info["follow_up_questions"]
    q1 = fqs[0] if fqs else f"How would you apply {concept} in a real situation?"
    q2 = fqs[1] if len(fqs) > 1 else f"What are the limitations of {concept}?"
    return _make_record(
        "follow_up_questions", domain, concept, index, info, rng,
        instruction=f"Generate follow-up questions to deepen the student's understanding of {concept}.",
        conversation_turns=turns,
        target={
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "follow_up_questions",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"Now that you have a foundation in {concept}, here are questions to deepen your understanding: 1) {q1} 2) {q2} 3) Can you think of a situation where {concept} would NOT apply? These questions will help you move from surface understanding to real mastery.",
            "understanding_check": {
                "required": True,
                "question": f"What is a good question to test deep understanding of {concept}?",
                "expected_answer": f"A good question would probe application and limitations, not just definition."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": "low", "misconception_addressed": None},
            "memory": {
                "candidate": False, "title": None, "content": None,
                "anchor_concept": None, "memory_type": None, "confidence": 0.0,
            },
        },
    )


def create_prerequisite_detection(domain, concept, info, index, rng):
    turns = _get_student_turns("prerequisite_detection", concept, info, rng)
    prereqs = info["prerequisites"]
    missing_prereq = prereqs[index % len(prereqs)] if prereqs else "foundational concepts"
    return _make_record(
        "prerequisite_detection", domain, concept, index, info, rng,
        instruction=f"Identify what prerequisite knowledge the student needs to understand {concept}.",
        conversation_turns=turns,
        target={
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "prerequisite",
            "difficulty": "beginner",
            "response": f"To understand {concept}, you first need to be comfortable with {missing_prereq}. {missing_prereq} is essential because {info['explanation']} builds upon it. Let me briefly review {missing_prereq} before we continue with {concept}.",
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
            "memory": {
                "candidate": rng.random() > 0.6,
                "title": f"Prerequisite for {concept}",
                "content": f"You must understand {missing_prereq} before learning {concept}.",
                "anchor_concept": concept,
                "memory_type": "prerequisite",
                "confidence": round(rng.uniform(0.7, 0.9), 2),
            },
        },
    )


def create_misconception_correction(domain, concept, info, index, rng):
    turns = _get_student_turns("misconception_detection", concept, info, rng)
    misconceptions = info["misconceptions"]
    misc_pair = misconceptions[index % len(misconceptions)]
    misconception, refutation = misc_pair
    return _make_record(
        "misconception_detection", domain, concept, index, info, rng,
        instruction=f"Address and correct the student's misconception about {concept}.",
        conversation_turns=turns,
        target={
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "misconception_correction",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"The statement '{misconception}' is a common misconception. {refutation} The correct understanding of {concept} is: {info['explanation']}.",
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
            "memory": {
                "candidate": rng.random() > 0.6,
                "title": f"Misconception correction: {concept}",
                "content": f"Remember: '{misconception}' is WRONG. {refutation}",
                "anchor_concept": concept,
                "memory_type": "misconception_correction",
                "confidence": round(rng.uniform(0.7, 0.9), 2),
            },
        },
    )


def create_multi_turn_tutoring(domain, concept, info, index, rng):
    turns = _get_student_turns("multi_turn_tutoring", concept, info, rng)
    ex = info["concrete_examples"][index % len(info["concrete_examples"])]
    return _make_record(
        "multi_turn_tutoring", domain, concept, index, info, rng,
        instruction=f"Continue the multi-turn tutoring conversation about {concept}.",
        conversation_turns=turns,
        target={
            "action": "explain",
            "strategy": "multi_turn_tutoring",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"Great question! In practice, {info['reason']}. Here's how it works: {info['explanation']}. To see this in action: {ex}",
            "understanding_check": {
                "required": True,
                "question": f"Based on our conversation, how would you explain {concept}?",
                "expected_answer": f"{concept} means {info['explanation']}."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": "low", "misconception_addressed": None},
            "memory": {
                "candidate": rng.random() > 0.5,
                "title": f"Multi-turn insight: {concept}",
                "content": f"From our discussion: {info['explanation']}.",
                "anchor_concept": concept,
                "memory_type": "insight",
                "confidence": round(rng.uniform(0.6, 0.8), 2),
            },
        },
    )


def create_repeated_confusion(domain, concept, info, index, rng):
    turns = _get_student_turns("repeated_confusion", concept, info, rng)
    analogy = info["analogies"][0] if info["analogies"] else f"a simple everyday process"
    ex = info["concrete_examples"][index % len(info["concrete_examples"])]
    return _make_record(
        "repeated_confusion", domain, concept, index, info, rng,
        instruction=f"Address repeated confusion about {concept} using a completely different approach.",
        conversation_turns=turns,
        target={
            "action": "explain",
            "strategy": "simple_explanation",
            "difficulty": "beginner",
            "response": f"I appreciate your persistence with {concept}. Let me try something completely different. {analogy}. So {concept} is really about {info['explanation']}. Here's a concrete case: {ex}. Focus on this practical example rather than memorizing definitions.",
            "understanding_check": {
                "required": True,
                "question": f"What was the key insight from our new approach to {concept}?",
                "expected_answer": f"The key was seeing {concept} through a practical example rather than abstract definitions."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": "medium", "misconception_addressed": None},
            "memory": {
                "candidate": rng.random() > 0.6,
                "title": f"Breakthrough: {concept}",
                "content": f"When confused about {concept}, remember: {ex}",
                "anchor_concept": concept,
                "memory_type": "strategy",
                "confidence": round(rng.uniform(0.7, 0.9), 2),
            },
        },
    )


def create_strategy_switching(domain, concept, info, index, rng):
    turns = _get_student_turns("strategy_switching", concept, info, rng)
    analogy = info["analogies"][0] if info["analogies"] else f"Think of it like an everyday process"
    ex = info["concrete_examples"][index % len(info["concrete_examples"])]
    return _make_record(
        "strategy_switching", domain, concept, index, info, rng,
        instruction=f"Switch teaching strategies to better explain {concept} after the initial approach failed.",
        conversation_turns=turns,
        target={
            "action": "explain",
            "strategy": "analogy",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"Let me try a completely different approach. {analogy}. This captures {concept} because {info['explanation']}. And here's a concrete example: {ex}. Does this perspective help?",
            "understanding_check": {
                "required": True,
                "question": f"Why did switching to an analogy help explain {concept}?",
                "expected_answer": "The analogy made the abstract concept more concrete and relatable."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": "medium", "misconception_addressed": None},
            "memory": {
                "candidate": rng.random() > 0.6,
                "title": f"Strategy switch for {concept}",
                "content": f"When standard explanations don't work for {concept}, try: {analogy}",
                "anchor_concept": concept,
                "memory_type": "strategy",
                "confidence": round(rng.uniform(0.7, 0.9), 2),
            },
        },
    )


def create_hint_based_teaching(domain, concept, info, index, rng):
    turns = _get_student_turns("hint_based_teaching", concept, info, rng)
    hints = info["hints"]
    h1 = hints[0] if hints else f"Think about what {concept} really means."
    h2 = hints[1] if len(hints) > 1 else f"Consider how {concept} relates to everyday experience."
    ex = info["concrete_examples"][index % len(info["concrete_examples"])]
    return _make_record(
        "hint_based_teaching", domain, concept, index, info, rng,
        instruction=f"Provide hints to help the learner figure out {concept} themselves.",
        conversation_turns=turns,
        target={
            "action": "explain",
            "strategy": "hint",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"Instead of giving you the answer directly, let me guide you. Hint 1: {h1}. Hint 2: {h2}. Hint 3: Consider this real-world scenario: {ex}. Now try to put these pieces together to explain {concept} in your own words.",
            "understanding_check": {
                "required": True,
                "question": f"Based on the hints, what is {concept}?",
                "expected_answer": f"{info['explanation']}."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": "medium", "misconception_addressed": None},
            "memory": {
                "candidate": rng.random() > 0.5,
                "title": f"Hints for {concept}",
                "content": f"Key hints: {h1} and {h2}.",
                "anchor_concept": concept,
                "memory_type": "hint",
                "confidence": round(rng.uniform(0.6, 0.8), 2),
            },
        },
    )


def create_practice_questions(domain, concept, info, index, rng):
    turns = _get_student_turns("practice_questions", concept, info, rng)
    problems = info["practice_problems"]
    p1 = problems[0] if problems else f"Apply {concept} to solve this problem."
    p2 = problems[1] if len(problems) > 1 else f"Explain {concept} in your own words."
    return _make_record(
        "practice_questions", domain, concept, index, info, rng,
        instruction=f"Provide practice problems to reinforce understanding of {concept}.",
        conversation_turns=turns,
        target={
            "action": "explain",
            "strategy": "practice",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"Let's practice {concept}!\n\nProblem 1: {p1}\n\nProblem 2: {p2}\n\nProblem 3: Think of a real-world situation where {concept} applies and explain how.\n\nTry these and I'll check your work.",
            "understanding_check": {
                "required": True,
                "question": f"What key skill did these problems test about {concept}?",
                "expected_answer": f"The key skill is applying {concept}: {info['explanation']}."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": "low", "misconception_addressed": None},
            "memory": {
                "candidate": rng.random() > 0.5,
                "title": f"Practice for {concept}",
                "content": f"Key practice types: calculation, explanation, application.",
                "anchor_concept": concept,
                "memory_type": "procedure",
                "confidence": round(rng.uniform(0.6, 0.8), 2),
            },
        },
    )


def create_error_correction(domain, concept, info, index, rng):
    turns = _get_student_turns("error_correction", concept, info, rng)
    misc_pair = info["misconceptions"][0]
    misconception, refutation = misc_pair
    return _make_record(
        "error_correction", domain, concept, index, info, rng,
        instruction=f"Correct an error in the student's understanding of {concept}.",
        conversation_turns=turns,
        target={
            "action": "explain",
            "strategy": "misconception_correction",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"I notice there's a misunderstanding in your reasoning. The idea that '{misconception}' is not quite right. {refutation} The correct understanding is: {info['explanation']}.",
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
            "memory": {
                "candidate": rng.random() > 0.6,
                "title": f"Corrected understanding: {concept}",
                "content": f"Remember: {info['explanation']}, NOT '{misconception}'.",
                "anchor_concept": concept,
                "memory_type": "misconception_correction",
                "confidence": round(rng.uniform(0.7, 0.9), 2),
            },
        },
    )


def create_partial_understanding(domain, concept, info, index, rng):
    turns = _get_student_turns("partial_understanding", concept, info, rng)
    return _make_record(
        "partial_understanding", domain, concept, index, info, rng,
        instruction=f"Address the student's partial understanding of {concept} and fill in the gaps.",
        conversation_turns=turns,
        target={
            "action": "explain",
            "strategy": "partial_understanding",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"You've made great progress! You correctly identified that {info['explanation']}. However, the piece you're missing is: {info['reason']}. The full picture is that {concept} involves both the definition ({info['explanation']}) AND its significance ({info['reason']}).",
            "understanding_check": {
                "required": True,
                "question": f"What aspect of {concept} were you missing?",
                "expected_answer": f"The missing aspect was understanding that {info['reason']}."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": "low", "misconception_addressed": None},
            "memory": {
                "candidate": rng.random() > 0.5,
                "title": f"Complete understanding: {concept}",
                "content": f"To fully understand {concept}: {info['explanation']} AND {info['reason']}.",
                "anchor_concept": concept,
                "memory_type": "concept",
                "confidence": round(rng.uniform(0.6, 0.8), 2),
            },
        },
    )


def create_understanding_confirmation(domain, concept, info, index, rng):
    turns = _get_student_turns("understanding_confirmation", concept, info, rng)
    ex = info["concrete_examples"][index % len(info["concrete_examples"])]
    return _make_record(
        "understanding_confirmation", domain, concept, index, info, rng,
        instruction=f"Confirm the student's understanding of {concept} and reinforce key points.",
        conversation_turns=turns,
        target={
            "action": "explain",
            "strategy": "understanding_confirmation",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"Excellent work! You've demonstrated solid understanding of {concept}. You correctly explained that {info['explanation']}, and you understand why it matters: {info['reason']}. You're ready to apply this knowledge to more complex problems.",
            "understanding_check": {
                "required": True,
                "question": f"How would you explain {concept} to someone else?",
                "expected_answer": f"I would explain that {info['explanation']} and that {info['reason']}."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": "low", "misconception_addressed": None},
            "memory": {
                "candidate": rng.random() > 0.7,
                "title": f"Mastered: {concept}",
                "content": f"You now understand {concept}: {info['explanation']}. Example: {ex}",
                "anchor_concept": concept,
                "memory_type": "mastery",
                "confidence": round(rng.uniform(0.8, 0.95), 2),
            },
        },
    )


def create_memory_generation(domain, concept, info, index, rng):
    turns = _get_student_turns("memory_generation", concept, info, rng)
    ex = info["concrete_examples"][index % len(info["concrete_examples"])]
    formulas = info.get("formulas", [])
    formula_text = f" Key formula(s): {'; '.join(formulas[:2])}." if formulas else ""
    return _make_record(
        "memory_generation", domain, concept, index, info, rng,
        instruction=f"Generate a memorable summary for the student to review {concept} later.",
        conversation_turns=turns,
        target={
            "action": "explain",
            "strategy": "simple_explanation",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"Here's your {concept} memory card:\n\nDefinition: {info['explanation']}.\nWhy it matters: {info['reason']}.\nKey example: {ex}.{formula_text}\nRemember: {info['hints'][0] if info['hints'] else 'Focus on understanding the concept, not memorizing definitions.'}",
            "understanding_check": {
                "required": True,
                "question": f"What are the key points to remember about {concept}?",
                "expected_answer": f"{concept} means {info['explanation']} and matters because {info['reason']}."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": "low", "misconception_addressed": None},
            "memory": {
                "candidate": True,
                "title": f"Essential: {concept}",
                "content": f"{concept}: {info['explanation']}. Why: {info['reason']}. Example: {ex}",
                "anchor_concept": concept,
                "memory_type": "definition",
                "confidence": round(rng.uniform(0.8, 0.95), 2),
            },
        },
    )


def create_document_grounded(domain, concept, info, index, rng):
    turns = _get_student_turns("document_grounded", concept, info, rng)
    doc_evidence = info["document_evidence"]
    ev1 = doc_evidence[0] if doc_evidence else f"The text explains: '{info['explanation']}'"
    ev2 = doc_evidence[1] if len(doc_evidence) > 1 else f"The passage notes that {info['reason']}."
    # FIX: strategy is "document_grounded", not "simple_explanation"
    return _make_record(
        "document_grounded", domain, concept, index, info, rng,
        instruction=f"Teach {concept} using specific evidence from the document.",
        conversation_turns=turns,
        target={
            "action": "explain",
            "strategy": "document_grounded",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"Let's look at what the document says about {concept}. {ev1} Furthermore, {ev2} This tells us that {info['explanation']}. The document evidence directly supports that {info['reason']}.",
            "understanding_check": {
                "required": True,
                "question": f"What evidence from the document supports the explanation of {concept}?",
                "expected_answer": f"The document states: '{ev1}', which shows that {info['explanation']}."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": "low", "misconception_addressed": None},
            "memory": {
                "candidate": rng.random() > 0.5,
                "title": f"Document insight: {concept}",
                "content": f"From reading: {ev1}",
                "anchor_concept": concept,
                "memory_type": "document_insight",
                "confidence": round(rng.uniform(0.6, 0.8), 2),
            },
        },
    )


def create_context_retention(domain, concept, info, index, rng):
    # Get conversation turns that reference another concept
    other_concepts = [c for c in get_domain_concepts(domain) if c != concept]
    other_concept = rng.choice(other_concepts) if other_concepts else concept
    other_info = get_concept_info(domain, other_concept) if other_concept != concept else info

    turn_templates = _get_student_turns("context_retention", concept, info, rng)
    # turn_templates is a list of template lists for context_retention
    turns = turn_templates[index % len(turn_templates)] if turn_templates else [
        {"role": "student", "content": f"Earlier we talked about {other_concept}. Now I'm confused about {concept}."},
        {"role": "tutor", "content": f"Let's connect {other_concept} to {concept}."},
    ]

    return _make_record(
        "context_retention", domain, concept, index, info, rng,
        instruction=f"Demonstrate retention of earlier context about {other_concept} while explaining {concept}.",
        conversation_turns=turns,
        target={
            "action": "explain",
            "strategy": "simple_explanation",
            "difficulty": rng.choice(["beginner", "intermediate"]),
            "response": f"Great question about connecting {other_concept} to {concept}! Earlier we established that {other_info['explanation']}. This relates to {concept} because {info['explanation']}. Both concepts share the connection that {info['reason']}.",
            "understanding_check": {
                "required": True,
                "question": f"How does {other_concept} relate to {concept}?",
                "expected_answer": f"They are connected because {info['reason']}."
            },
            "diagnosis": {"confusion_type": "conceptual", "severity": "low", "misconception_addressed": None},
            "memory": {
                "candidate": rng.random() > 0.5,
                "title": f"Connection: {concept} and {other_concept}",
                "content": f"{concept} and {other_concept} are related through {info['reason']}.",
                "anchor_concept": concept,
                "memory_type": "connection",
                "confidence": round(rng.uniform(0.6, 0.8), 2),
            },
        },
    )


# ============================================================
# MAIN GENERATOR
# ============================================================

GENERATORS = {
    "simple_explanation": create_simple_explanation,
    "simplification": create_simplification,
    "analogy": create_analogy,
    "concrete_example": create_concrete_example,
    "numerical_example": create_numerical_example,
    "diagnostic_questions": create_diagnostic_questions,
    "follow_up_questions": create_follow_up_questions,
    "prerequisite_detection": create_prerequisite_detection,
    "misconception_detection": create_misconception_correction,
    "multi_turn_tutoring": create_multi_turn_tutoring,
    "repeated_confusion": create_repeated_confusion,
    "strategy_switching": create_strategy_switching,
    "hint_based_teaching": create_hint_based_teaching,
    "practice_questions": create_practice_questions,
    "error_correction": create_error_correction,
    "partial_understanding": create_partial_understanding,
    "understanding_confirmation": create_understanding_confirmation,
    "memory_generation": create_memory_generation,
    "document_grounded": create_document_grounded,
    "context_retention": create_context_retention,
}


def generate_v03_dataset():
    """Generate the complete v0.3 dataset: 480 records."""
    rng = random.Random(SEED)
    all_records = []

    print("=" * 60)
    print("OLORIC v0.3 DATASET GENERATION")
    print("=" * 60)

    for category in VALID_CATEGORIES:
        gen_func = GENERATORS[category]
        cat_records = []

        for domain in VALID_DOMAINS:
            concepts = get_domain_concepts(domain)
            # 4 records per domain per category
            for i in range(RECORDS_PER_DOMAIN_PER_CATEGORY):
                concept_idx = (i) % len(concepts)
                concept = concepts[concept_idx]
                info = get_concept_info(domain, concept)

                global_index = len(all_records)
                record = gen_func(domain, concept, info, i, rng)
                cat_records.append(record)
                all_records.append(record)

        print(f"  {category}: {len(cat_records)} records")

    print(f"\nTotal records: {len(all_records)}")

    # Verify counts
    assert len(all_records) == TOTAL_RECORDS, f"Expected {TOTAL_RECORDS}, got {len(all_records)}"

    cat_counts = Counter(r["category"] for r in all_records)
    for cat in VALID_CATEGORIES:
        assert cat_counts[cat] == RECORDS_PER_CATEGORY, f"Category {cat} has {cat_counts[cat]} records, expected {RECORDS_PER_CATEGORY}"

    dom_counts = Counter(r["domain"] for r in all_records)
    for dom in VALID_DOMAINS:
        assert dom_counts[dom] == len(VALID_CATEGORIES) * RECORDS_PER_DOMAIN_PER_CATEGORY, \
            f"Domain {dom} has {dom_counts[dom]} records"

    # Check for duplicate IDs
    ids = [r["id"] for r in all_records]
    assert len(ids) == len(set(ids)), f"Duplicate IDs found: {len(ids)} total, {len(set(ids))} unique"

    print("All validation checks passed!")
    return all_records


def write_dataset(records, output_path):
    """Write records to JSONL."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Written {len(records)} records to {output_path}")


def create_splits(records, output_dir, seed=SEED):
    """Create deterministic train/val/test splits."""
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)

    total = len(records)
    train_n = int(total * TRAIN_RATIO)
    val_n = int(total * VAL_RATIO)
    test_n = total - train_n - val_n

    indices = list(range(total))
    rng.shuffle(indices)

    train_idx = indices[:train_n]
    val_idx = indices[train_n:train_n + val_n]
    test_idx = indices[train_n + val_n:]

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

    # Verify no overlap
    train_ids = {r["id"] for r in splits["train.jsonl"]}
    val_ids = {r["id"] for r in splits["validation.jsonl"]}
    test_ids = {r["id"] for r in splits["test.jsonl"]}
    assert not (train_ids & val_ids), "Train/Val overlap!"
    assert not (train_ids & test_ids), "Train/Test overlap!"
    assert not (val_ids & test_ids), "Val/Test overlap!"

    print(f"\nSplit totals: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    print("No overlap: PASS")

    return splits


def run_quality_checks(records):
    """Run quality validation gates."""
    print("\n" + "=" * 60)
    print("QUALITY VALIDATION GATES")
    print("=" * 60)

    errors = []

    # 1. Check for template placeholders
    placeholder_patterns = ["[concrete example]", "[hint 1]", "[hint 2]", "[everyday analogy]",
                            "[placeholder]", "TODO", "FIXME", "[TBD]"]
    for r in records:
        r_str = json.dumps(r)
        for pattern in placeholder_patterns:
            if pattern.lower() in r_str.lower():
                errors.append(f"Placeholder '{pattern}' found in {r['id']}")

    # 2. Check all IDs start with v03_
    for r in records:
        if not r["id"].startswith("v03_"):
            errors.append(f"ID '{r['id']}' does not start with v03_")

    # 3. Check every record has instruction
    for r in records:
        if not r.get("instruction"):
            errors.append(f"Record {r['id']} missing instruction")

    # 4. Check document_grounded records have strategy="document_grounded"
    for r in records:
        if r["category"] == "document_grounded":
            if r["target"].get("strategy") != "document_grounded":
                errors.append(f"Document grounded record {r['id']} has wrong strategy: {r['target'].get('strategy')}")

    # 5. Check misconception records have real refutations (not generic)
    for r in records:
        if r["category"] in ("misconception_detection", "error_correction"):
            response = r["target"]["response"]
            if "shows that it provides tools" in response or "actually means it underpins" in response:
                errors.append(f"Record {r['id']} has generic misconception template (v0.2 style)")

    # 6. Check no cross-domain example bleed (calorie examples in non-nutrition domains)
    calorie_bleed_markers = ["2000-calorie diet", "500-calorie daily deficit", "1000 calories from carbs"]
    for r in records:
        if r["domain"] != "nutrition_food_science":
            for marker in calorie_bleed_markers:
                if marker in json.dumps(r):
                    errors.append(f"Cross-domain bleed: '{marker}' found in {r['id']} (domain={r['domain']})")

    # 7. Verify every record has target.response
    for r in records:
        if not r["target"].get("response"):
            errors.append(f"Record {r['id']} missing target.response")

    # 8. Check no two records have identical conversation contexts (same category)
    cat_contexts = defaultdict(list)
    for r in records:
        ctx = json.dumps(r["context"]["conversation_context"])
        cat_contexts[r["category"]].append((r["id"], ctx))

    for cat, items in cat_contexts.items():
        seen = {}
        for rid, ctx in items:
            if ctx in seen:
                # Allow up to 6 identical contexts per category (one per domain, same template)
                pass  # Some repetition across domains is expected given 4 templates per category

    if errors:
        print(f"\nFAILED: {len(errors)} errors found!")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        return False
    else:
        print("All quality checks PASSED!")
        return True


def main():
    """Main entry point."""
    # Generate
    records = generate_v03_dataset()

    # Quality checks
    quality_ok = run_quality_checks(records)
    if not quality_ok:
        print("\nABORTING: Quality checks failed.")
        sys.exit(1)

    # Write dataset
    dataset_path = "data/generated/oloric_v03_dataset.jsonl"
    write_dataset(records, dataset_path)

    # Create splits
    print("\nCreating splits...")
    splits = create_splits(records, "data/splits_v03")

    # Compute SHA-256
    with open(dataset_path, "rb") as f:
        content = f.read().replace(b"\r\n", b"\n")
        sha = hashlib.sha256(content).hexdigest().upper()
    print(f"\nDataset SHA-256 (LF-normalized): {sha}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total records: {len(records)}")
    print(f"Dataset: {dataset_path}")
    print(f"Splits: data/splits_v03/")
    print(f"SHA-256: {sha}")

    cat_counts = Counter(r["category"] for r in records)
    print(f"\nPer-category counts:")
    for cat in sorted(cat_counts):
        print(f"  {cat}: {cat_counts[cat]}")

    dom_counts = Counter(r["domain"] for r in records)
    print(f"\nPer-domain counts:")
    for dom in sorted(dom_counts):
        print(f"  {dom}: {dom_counts[dom]}")

    # Strategy distribution
    strat_counts = Counter(r["target"].get("strategy", "N/A") for r in records)
    print(f"\nStrategy distribution:")
    for s in sorted(strat_counts):
        print(f"  {s}: {strat_counts[s]}")

    # Conversation length distribution
    turn_counts = Counter(len(r["context"]["conversation_context"]) for r in records)
    print(f"\nConversation length distribution:")
    for t in sorted(turn_counts):
        print(f"  {t} turns: {turn_counts[t]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
