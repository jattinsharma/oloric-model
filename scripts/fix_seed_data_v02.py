#!/usr/bin/env python3
"""
Build the v0.2 corrected dataset WITHOUT touching v0.1 artifacts.

Strategy: reuse scripts/generate_seed_data_improved.py's content-filled
create_* functions (deterministic via seeded random), drive them directly at
4 examples per category x 6 domains = 480 (matching v0.1's exact balance),
patch its known content defects at generation time, then write
data/generated/oloric_v02_dataset.jsonl.

Defects patched (monkeypatched, upstream file left unmodified):
  1. create_simplification emits a literal '[everyday analogy]' slot.
  2. get_concept_explanation() returns the domain's first formula regardless
     of concept (concept-incorrect explanations like "GDP is defined by MPC
     formula"). We provide a per-concept explanation table with a safe
     domain-level fallback, plus factually correct reasons per concept.

Determinism: random.Random(42) controls every random choice.
No training is triggered; no v0.1 file is modified.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import generate_seed_data_improved as gi  # noqa: E402

OUT_PATH = os.path.join("data", "generated", "oloric_v02_dataset.jsonl")
SEED = 42
PER_CATEGORY_PER_DOMAIN = 4  # 4 * 20 categories * 6 domains = 480

DOMAINS = gi.VALID_DOMAINS
CATEGORIES = gi.VALID_CATEGORIES

# ---------------------------------------------------------------- content tables
# Per-concept explanations (concept -> correct, concept-specific definition).
CONCEPT_EXPLANATIONS = {
    # economics
    "MPC (Marginal Propensity to Consume)": "the fraction of each extra dollar of income that a household spends rather than saves",
    "GDP": "the total market value of all final goods and services produced within a country in a given period",
    "Inflation": "a sustained rise in the general price level, which reduces purchasing power over time",
    "Supply and Demand": "the model describing how prices emerge from producers' willingness to sell and consumers' willingness to buy",
    "Opportunity Cost": "the value of the best alternative you give up when making a choice",
    "Elasticity": "the responsiveness of quantity demanded or supplied to a change in price",
    "Comparative Advantage": "the ability to produce a good at a lower opportunity cost than another producer",
    "Fiscal Policy": "the government's use of spending and taxation to influence the economy",
    "Monetary Policy": "the central bank's management of interest rates and money supply to steer growth and inflation",
    "Externalities": "costs or benefits of an activity that fall on third parties, like pollution or education spillovers",
    # accountancy
    "Double-Entry Bookkeeping": "the recording system where every transaction is entered as equal debits and credits, keeping Assets = Liabilities + Equity in balance",
    "Accrual Accounting": "recording revenue when earned and expenses when incurred, regardless of when cash moves",
    "Financial Ratios": "standardized comparisons of financial figures, like the current ratio, that summarize a firm's health",
    "Depreciation Methods": "systematic ways of spreading an asset's cost over its useful life, such as straight-line depreciation",
    "Cost-Volume-Profit Analysis": "the study of how costs, sales volume, and profit interact, including the break-even point",
    "Budget Variance Analysis": "comparing actual results to budgeted amounts and explaining the differences",
    "Time Value of Money": "the principle that a dollar today is worth more than a dollar later because it can earn interest",
    "Inventory Valuation": "assigning cost to unsold goods using methods like FIFO or weighted average",
    "Tax Principles": "the rules determining how taxable income is computed and tax obligations arise",
    "Audit Procedures": "the verification steps auditors use, such as inspecting records and confirming balances, to support an opinion",
    # mathematics
    "Quadratic Equations": "equations of the form ax² + bx + c = 0, solved with factoring, completing the square, or the quadratic formula",
    "Derivatives": "measures of instantaneous rate of change, defined as the limit of the average rate of change",
    "Integrals": "accumulations of quantity under a curve, the reverse operation of differentiation",
    "Probability Distributions": "functions assigning probabilities to each possible outcome of a random experiment",
    "Linear Algebra": "the study of vectors, matrices, and linear transformations, central to solving systems of equations",
    "Trigonometric Identities": "equalities like sin²θ + cos²θ = 1 that hold for all angle values",
    "Logarithms": "the inverse operation of exponentiation, answering 'to what power must the base be raised?'",
    "Exponential Functions": "functions of the form f(x) = a·bˣ where the rate of change is proportional to the value",
    "Matrix Operations": "rules for adding, scaling, and multiplying rectangular arrays of numbers, with dimension-matching conditions",
    "Statistical Inference": "drawing conclusions about a population from a sample, using estimation and hypothesis testing",
    # science
    "Photosynthesis": "the process by which plants convert light energy, water, and CO₂ into glucose and oxygen",
    "Newton's Laws": "the three laws of motion describing inertia, F = ma, and action-reaction pairs",
    "Atomic Structure": "the organization of protons, neutrons, and electrons within an atom",
    "Chemical Bonding": "the forces, ionic or covalent, that hold atoms together in compounds",
    "DNA Replication": "the semi-conservative copying of DNA before cell division, with A-T and G-C base pairing",
    "Cell Mitosis": "nuclear division producing two genetically identical daughter cells",
    "Thermodynamics": "the laws governing energy transfer, heat flow, and entropy",
    "Electromagnetic Spectrum": "the full range of electromagnetic waves by wavelength, from radio to gamma rays",
    "Plate Tectonics": "the theory that Earth's lithosphere is divided into moving plates causing earthquakes and mountains",
    "Natural Selection": "the mechanism where heritable traits that improve survival become more common over generations",
    # nutrition_food_science
    "Macronutrients": "the nutrients needed in large amounts — carbohydrates, proteins, and fats — that supply energy",
    "Glycemic Index": "a ranking of how quickly carbohydrate-containing foods raise blood glucose",
    "Protein Synthesis": "the process where cells use DNA instructions, via mRNA and ribosomes, to build proteins",
    "Vitamin Absorption": "how vitamins are taken up, notably that fat-soluble vitamins A, D, E, and K require dietary fat",
    "Food Preservation": "methods like refrigeration, drying, and canning that slow or stop microbial growth to keep food safe",
    "Enzyme Activity": "the rate at which enzymes catalyze reactions, sensitive to temperature and pH",
    "Metabolism": "the sum of all chemical reactions that sustain life, catabolic and anabolic",
    "Nutrient Deficiencies": "health problems arising from insufficient intake of essential nutrients, like scurvy from lack of vitamin C",
    "Food Safety": "practices that prevent contamination and foodborne illness, from handwashing to safe storage",
    "Dietary Fiber": "indigestible plant carbohydrate that aids digestion; soluble fiber helps lower cholesterol",
    # general_academic
    "Critical Thinking": "the disciplined evaluation of claims and arguments using evidence and logic",
    "Research Methods": "systematic procedures, like experiments and surveys, for gathering and analyzing data",
    "Argument Structure": "the organization of an argument into claim, supporting premises, and conclusion",
    "Evidence Evaluation": "judging the quality, relevance, and sufficiency of information supporting a claim",
    "Logical Fallacies": "common reasoning errors, like post hoc or bandwagon, that undermine arguments",
    "Study Design": "the plan for a study — sampling, controls, and measures — that determines what it can show",
    "Bias Identification": "spotting one-sided influence on information or reasoning, such as confirmation bias",
    "Effective Communication": "conveying ideas clearly and engagingly through structure, language, and delivery",
    "Learning Strategies": "techniques like spaced repetition and active recall that improve retention and understanding",
    "Information Synthesis": "combining insights from multiple sources into a coherent, complete picture",
}

# Per-concept importance reasons (fall back to domain-level reason).
CONCEPT_REASONS = {
    "Food Preservation": "preventing spoilage keeps nutrients intact and food safe to eat",
    "Protein Synthesis": "proteins drive every repair and growth process in the body",
    "Enzyme Activity": "digestion and every biochemical reaction in the body depend on enzyme speed",
    "Monetary Policy": "interest-rate decisions shape borrowing, spending, and inflation for everyone",
    "MPC (Marginal Propensity to Consume)": "the multiplier effect that amplifies fiscal policy runs on MPC",
    "Audit Procedures": "reliable audits depend on thorough verification, not just trusting the numbers",
    "Double-Entry Bookkeeping": "the debit-credit balance is what makes errors detectable and statements trustworthy",
    "Matrix Operations": "computer graphics, statistics, and engineering all reduce to matrix computations",
    "Cell Mitosis": "growth and wound healing depend on cells dividing identically",
    "Photosynthesis": "nearly all food chains and the oxygen we breathe originate in photosynthesis",
    "Dietary Fiber": "fiber intake affects digestion, blood sugar, and cholesterol levels",
    "Argument Structure": "clear reasoning requires knowing what supports what",
    "Bias Identification": "unbiased reading of sources is the basis of sound judgment",
    "Derivatives": "optimization problems across science and economics are solved with derivatives",
}

DOMAIN_REASON = gi.get_concept_reason  # domain-level fallback stays as-is


def explain(domain: str, concept: str) -> str:
    """Concept-specific explanation with domain fallback (patched replacement)."""
    exp = CONCEPT_EXPLANATIONS.get(concept)
    if exp:
        return exp
    formulas = gi.DOMAIN_CONTENT.get(domain, {}).get("formulas") or []
    if formulas:
        return f"defined for this course by {formulas[0]}"
    return f"an important concept in {domain}"


def reason(domain: str, concept: str) -> str:
    """Concept-specific importance reason with domain fallback (patched)."""
    if concept in CONCEPT_REASONS:
        return CONCEPT_REASONS[concept]
    return DOMAIN_REASON(domain, concept)


def analogy_for(domain: str, concept: str) -> str:
    """Deterministic analogy from the improved generator's own table (patch #1)."""
    table = gi.create_analogy.__globals__.get("analogies", {})
    lst = table.get(domain) or ["a useful tool for reasoning"]
    # pick stably by concept so output is deterministic
    idx = sum(ord(ch) for ch in concept) % len(lst)
    return lst[idx]


def patched_get_concept_explanation(domain, concept):
    return explain(domain, concept)


def patched_get_concept_reason(domain, concept):
    return reason(domain, concept)


def patched_create_simplification(domain, index):
    """Re-implementation without the '[everyday analogy]' slot."""
    import random as _r
    domain_info = gi.DOMAIN_CONTENT[domain]
    concept = _r.choice(domain_info["concepts"])
    formula = _r.choice(domain_info["formulas"]) if domain_info["formulas"] else f"the principle of {concept}"
    exp = explain(domain, concept)
    rs = reason(domain, concept)
    # deterministic simple everyday comparison, concept-driven
    simple = {
        "economics": "a household budget",
        "accountancy": "a scoreboard for money",
        "mathematics": "a set of rules for a board game",
        "science": "a recipe with exact steps",
        "nutrition_food_science": "a fuel gauge for the body",
        "general_academic": "a toolbox of thinking skills",
    }[domain]
    return {
        "id": gi.generate_example_id("simplification", domain, index),
        "category": "simplification",
        "domain": domain,
        "instruction": f"Simplify the explanation of {concept} for a beginner.",
        "context": gi.create_base_context(domain, concept, "beginner"),
        "target": {
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "simplification",
            "difficulty": "beginner",
            "response": (
                f"Think of {concept} like {simple}. Basically, it means {exp}, "
                f"instead of the technical definition involving {formula}. In short: {rs}."
            ),
            "understanding_check": {
                "required": True,
                "question": f"In your own words, what does {concept} mean?",
                "expected_answer": f"It means {exp}.",
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "medium",
                "misconception_addressed": "Belief that complex terminology is necessary for understanding",
            },
            "memory": {
                "candidate": _r.random() > 0.6,
                "title": f"Simple Definition: {concept}",
                "content": f"{concept} simply means {exp}.",
                "anchor_concept": concept,
                "memory_type": "definition",
                "confidence": round(_r.uniform(0.7, 0.9), 2),
            },
        },
    }


def patched_create_diagnostic_questions(domain, index):
    """Concept-specific, index-varied diagnostic questions.

    Replaces the improved generator's generic 4-question template, which is
    verbatim-identical to several benchmark targets (verified leakage).
    Four rotating variants probe different confusion modes: definitional,
    orientation, misconception/confidence, and application/prerequisites.
    """
    import random as _r
    domain_info = gi.DOMAIN_CONTENT[domain]
    concept = _r.choice(domain_info["concepts"])
    prereqs = domain_info["prerequisites"]
    p1 = prereqs[index % len(prereqs)]
    p2 = prereqs[(index + 1) % len(prereqs)]
    exp = explain(domain, concept)

    if index % 4 == 0:
        resp = (
            f"To pinpoint where the confusion about {concept} is, let me ask: "
            f"1) In your own words, what does {concept} mean so far? "
            f"2) Which feels more unclear: the definition of {concept}, or how to apply it? "
            f"3) Have you worked with {p1} before, since {concept} builds on it? "
            f"4) Can you describe one example of {concept} you remember?"
        )
    elif index % 4 == 1:
        resp = (
            f"Before I explain {concept} again, help me understand your starting point: "
            f"1) What do you currently believe {concept} means? "
            f"2) Do you remember the definition, an example, or neither? "
            f"3) If I said {concept} means {exp}, would that sound familiar? "
            f"4) What made you decide you were confused about {concept}?"
        )
    elif index % 4 == 2:
        resp = (
            f"Let me find out what kind of confusion we're dealing with about {concept}: "
            f"1) On a scale of 1-5, how confident are you that you could define {concept}? "
            f"2) Where did you first learn about {concept}, and what stuck? "
            f"3) Does {p2} seem connected to {concept} to you, or unrelated? "
            f"4) If you had to guess, what would you say {concept} is used for?"
        )
    else:
        resp = (
            f"To see whether the gap is knowledge or application, answer what you can about {concept}: "
            f"1) Have you tried any {domain} problems involving {concept}? "
            f"2) Which prerequisite feels shakier right now: {p1} or {p2}? "
            f"3) If you had to explain {concept} in one sentence, what would you say? "
            f"4) What did you expect {concept} to be about before we started?"
        )

    return {
        "id": gi.generate_example_id("diagnostic_questions", domain, index),
        "category": "diagnostic_questions",
        "domain": domain,
        "instruction": f"Ask diagnostic questions to identify the nature of confusion about {concept}.",
        "context": gi.create_base_context(domain, concept),
        "target": {
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "diagnostic_questions",
            "difficulty": "beginner",
            "response": resp,
            "understanding_check": {
                "required": False,
                "question": None,
                "expected_answer": None,
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "medium",
                "misconception_addressed": None,
            },
            "memory": {
                "candidate": False,
                "title": None,
                "content": None,
                "anchor_concept": None,
                "memory_type": None,
                "confidence": 0.0,
            },
        },
    }


def main():
    rng = random.Random(SEED)
    random.seed(SEED)  # the create_* functions call random.* directly

    # ---- monkeypatch defects in the improved generator (upstream untouched) ----
    gi.get_concept_explanation = patched_get_concept_explanation
    gi.get_concept_reason = patched_get_concept_reason
    gi.create_simplification = patched_create_simplification
    gi.create_diagnostic_questions = patched_create_diagnostic_questions
    # make get_concept_example deterministic too
    _orig_example = gi.get_concept_example

    def det_get_concept_example(domain, concept):
        pairs = gi.DOMAIN_CONTENT[domain]["examples"]
        idx = sum(ord(ch) for ch in concept) % len(pairs)
        return pairs[idx][0]

    gi.get_concept_example = det_get_concept_example

    # ---- drive the create_* functions directly: 4 x 20 categories x 6 domains ----
    all_examples = []
    for domain in DOMAINS:
        for category in CATEGORIES:
            creator = getattr(gi, f"create_{category}", None)
            if creator is None:
                # a few function names differ from category names
                alt = {
                    "misconception_detection": gi.create_misconception_correction,
                    "document_grounded": gi.create_document_grounded_teaching,
                }
                creator = alt.get(category)
            if creator is None:
                raise RuntimeError(f"No generator function for category {category}")
            for i in range(PER_CATEGORY_PER_DOMAIN):
                ex = creator(domain, i)
                all_examples.append(ex)

    assert len(all_examples) == 480, f"expected 480, got {len(all_examples)}"

    # ---- post-pass: namespace IDs to guarantee disjointness from v0.1 and the ----
    # benchmark ID spaces (the benchmark reuses the {domain}_{category}_NNN scheme,
    # so bare IDs can collide, e.g. *_001/_003). Prefix keeps them clearly separate.
    for ex in all_examples:
        ex["id"] = "v02_" + ex["id"]
    ids = [e["id"] for e in all_examples]
    assert len(set(ids)) == 480, "duplicate IDs detected"

    # deterministic shuffle (materialize_splits uses the same style)
    rng.shuffle(all_examples)

    # ---- fix memory-shape inconsistencies deterministically ----
    for ex in all_examples:
        mem = ex["target"].get("memory", {})
        if mem.get("candidate"):
            mem.setdefault("title", None)
            if not mem.get("title"):
                mem["title"] = f"Key: {ex['target'].get('anchor_concept') or ex['context']['learner_state']['concept']}"
            if not mem.get("content"):
                concept = ex["context"]["candidate_concept"] if "candidate_concept" in ex["context"] else ex["context"]["learner_state"]["concept"]
                mem["content"] = f"{concept} — {explain(ex['domain'], concept)}."
            if not mem.get("anchor_concept"):
                mem["anchor_concept"] = ex["context"]["learner_state"]["concept"]
        else:
            # candidate=False: clear filled memory fields for shape consistency
            for k in ("title", "content", "anchor_concept", "memory_type"):
                mem[k] = None
            mem["confidence"] = 0.0

    # also fix understanding_check.required=True without question (rare)
    for ex in all_examples:
        uc = ex["target"].get("understanding_check", {})
        if uc.get("required") and not uc.get("question"):
            concept = ex["context"]["learner_state"]["concept"]
            uc["question"] = f"What is the main idea of {concept}?"
            uc["expected_answer"] = uc.get("expected_answer") or f"The main idea is that {explain(ex['domain'], concept)}."

    # ---- write output (LF endings, do not touch v0.1 files) ----
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_examples)} examples -> {OUT_PATH}")

    # quick placeholder self-check using the strict validator
    import validate_dataset_content as vc

    passed, errors, warnings, stats = vc.validate_file(OUT_PATH)
    print(f"Self-check placeholder occurrences: {stats['placeholder_occurrences']}")
    print(f"Self-check errors: {len(errors)}  warnings: {len(warnings)}")
    if errors[:10]:
        for e in errors[:10]:
            print("  X", e)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
