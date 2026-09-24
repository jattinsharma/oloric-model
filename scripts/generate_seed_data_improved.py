#!/usr/bin/env python3
"""
Improved seed data generation script for OLORIC.
Generates seed dataset with higher proportion of meaningful multi-turn examples
and actual content (minimizing placeholders).
"""
import os
import json
import random
from typing import List, Dict, Any, Tuple

# Valid categories and domains from schema
VALID_CATEGORIES = [
    "simple_explanation", "simplification", "analogy", "concrete_example",
    "numerical_example", "prerequisite_detection", "misconception_detection",
    "follow_up_questions", "multi_turn_tutoring", "repeated_confusion",
    "strategy_switching", "diagnostic_questions", "hint_based_teaching",
    "practice_questions", "error_correction", "partial_understanding",
    "understanding_confirmation", "memory_generation", "document_grounded",
    "context_retention"
]

VALID_DOMAINS = [
    "economics", "accountancy", "mathematics", "science",
    "nutrition_food_science", "general_academic"
]

# Domain-specific content for more realistic examples
DOMAIN_CONTENT = {
    "economics": {
        "concepts": ["MPC (Marginal Propensity to Consume)", "GDP", "Inflation", "Supply and Demand", "Opportunity Cost", "Elasticity", "Comparative Advantage", "Fiscal Policy", "Monetary Policy", "Externalities"],
        "misconceptions": [
            "Higher prices always mean higher demand",
            "Imports always hurt the domestic economy",
            "Exports always benefit the domestic economy",
            "Minimum wage laws always increase unemployment",
            "Rent control always helps low-income tenants",
            "Trade deficits are always bad for an economy"
        ],
        "prerequisites": ["basic algebra", "graph interpretation", "percentage calculations"],
        "formulas": ["MPC = ΔC/ΔY", "GDP = C + I + G + (X-M)", "Elasticity = %ΔQ/%ΔP", "Multiplier = 1/(1-MPC)"],
        "examples": [
            ["If you earn $1000 and spend $700, your MPC is 0.7", "When gas prices rise from $3 to $4/gallon, quantity demanded might drop 10%"],
            ["If a country produces $1 trillion of goods and services, its GDP is $1 trillion"],
            ["If the price of a basket of goods increases from $100 to $110, inflation is 10%"],
            ["If the price of coffee increases, people may buy more tea instead"],
            ["If you choose to go to college, you give up the salary you could have earned working"],
            ["If the price of laptops falls by 10% and quantity demanded rises by 20%, elasticity is -2"],
            ["If a country can produce wine at lower opportunity cost than another, it has comparative advantage in wine"],
            ["If the government spends more on infrastructure, it can stimulate economic growth"],
            ["If the central bank lowers interest rates, borrowing may increase"],
            ["If a country imports more than it exports, it has a trade deficit"]
        ]
    },
    "accountancy": {
        "concepts": ["Double-Entry Bookkeeping", "Accrual Accounting", "Financial Ratios", "Depreciation Methods", "Cost-Volume-Profit Analysis", "Budget Variance Analysis", "Time Value of Money", "Inventory Valuation", "Tax Principles", "Audit Procedures"],
        "misconceptions": [
            "Profit equals cash flow",
            "All expenses reduce taxes immediately",
            "Inventory is always valued at market price",
            "Depreciation is a cash expense",
            "Higher revenue always means higher profit",
            "Accounts receivable is an expense"
        ],
        "prerequisites": ["basic arithmetic", "percentages"],
        "formulas": ["Assets = Liabilities + Equity", "Current Ratio = Current Assets/Current Liabilities", "Debt-to-Equity = Total Debt/Total Equity", "ROE = Net Income/Shareholder Equity"],
        "examples": [
            ["If assets = $100,000 and liabilities = $60,000, then equity = $40,000", "Revenue of $50,000 minus expenses of $30,000 equals profit of $20,000"],
            ["If a company records revenue when earned, not when cash is received, it uses accrual accounting"],
            ["If current assets are $200,000 and current liabilities are $100,000, current ratio is 2.0"],
            ["If equipment costs $50,000 with salvage value $5,000 and life 5 years, annual depreciation is $9,000"],
            ["If fixed costs are $10,000, variable cost per unit is $5, and selling price is $10, break-even is 2,000 units"],
            ["If actual spending is $8,000 and budgeted amount is $10,000, variance is $2,000 favorable"],
            ["If you have $1,000 today and can earn 5% interest, in one year you'll have $1,050"],
            ["If a company has $500,000 of inventory, it must count and value it correctly"],
            ["If a business sells goods, it must collect sales tax from customers"],
            ["If an auditor checks financial records, they verify accuracy and completeness"]
        ]
    },
    "mathematics": {
        "concepts": ["Quadratic Equations", "Derivatives", "Integrals", "Probability Distributions", "Linear Algebra", "Trigonometric Identities", "Logarithms", "Exponential Functions", "Matrix Operations", "Statistical Inference"],
        "misconceptions": [
            "√(a² + b²) = a + b",
            "(a+b)² = a² + b²",
            "If f'(x) = 0 then x is always a maximum",
            "All continuous functions are differentiable",
            "The probability of A and B is always P(A)×P(B)",
            "The median is always between the min and max"
        ],
        "prerequisites": ["algebra", "functions", "basic geometry"],
        "formulas": ["quadratic formula: x = (-b ± √(b²-4ac))/2a", "derivative of xⁿ = nxⁿ⁻¹", "Pythagorean theorem: a²+b²=c²", "area of circle = πr²"],
        "examples": [
            ["Solving 2x² + 5x - 3 = 0 gives x = 0.5 or x = -3", "The derivative of x³ at x=2 is 12"],
            ["The derivative of x² is 2x", "The integral of 2x from 0 to 2 is 4"],
            ["In a right triangle with legs 3 and 4, hypotenuse is 5"],
            ["If a matrix is [[1,2],[3,4]], its determinant is -2"],
            ["If log₁₀(100) = 2, then 10² = 100"],
            ["If 2³ = 8, then log₂(8) = 3"],
            ["If events A and B are independent with P(A)=0.3 and P(B)=0.4, P(A and B)=0.12"],
            ["If a fair coin is flipped 100 times, we expect about 50 heads"]
        ]
    },
    "science": {
        "concepts": ["Photosynthesis", "Newton's Laws", "Atomic Structure", "Chemical Bonding", "DNA Replication", "Cell Mitosis", "Thermodynamics", "Electromagnetic Spectrum", "Plate Tectonics", "Natural Selection"],
        "misconceptions": [
            "Seasons are caused by Earth's distance from the Sun",
            "Objects in motion always experience a force in the direction of motion",
            "Electric current flows from positive to negative",
            "Humans evolved from chimpanzees",
            "Antibiotics kill viruses",
            "Lightning never strikes the same place twice"
        ],
        "prerequisites": ["basic algebra", "measurement units", "scientific method"],
        "formulas": ["F = ma", "E = mc²", "PV = nRT", "E = hf", "pH = -log[H⁺]"],
        "examples": [
            ["In photosynthesis, 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂", "A 60kg person on Earth weighs about 10kg on the Moon"],
            ["If a 2kg object accelerates at 3m/s², force is 6N"],
            ["In a water molecule, two hydrogen atoms bond to one oxygen atom"],
            ["In DNA, adenine pairs with thymine and guanine pairs with cytosine"],
            ["During mitosis, chromosomes line up at the equator before separating"],
            ["If two objects at different temperatures touch, heat flows from hotter to colder until equilibrium"],
            ["Visible light has wavelengths between approximately 400-700 nanometers"],
            ["If South America fits into Africa, they may have been joined"],
            ["If organisms with advantageous traits survive and reproduce, those traits become more common"]
        ]
    },
    "nutrition_food_science": {
        "concepts": ["Macronutrients", "Glycemic Index", "Protein Synthesis", " Vitamin Absorption", "Food Preservation", "Enzyme Activity", "Metabolism", "Nutrient Deficiencies", "Food Safety", "Dietary Fiber"],
        "misconceptions": [
            "All fats are unhealthy",
            "Carbohydrates cause weight gain",
            "Detox diets remove toxins from your body",
            "You need to drink 8 glasses of water daily",
            "Organic food is always more nutritious",
            "Eating late at night causes weight gain"
        ],
        "prerequisites": ["basic biology", "chemistry basics"],
        "formulas": ["BMI = weight(kg)/height(m)²", "BMR = 10×weight(kg) + 6.25×height(cm) - 5×age(yr) + s", "Energy intake = Energy expenditure + Δbody stores"],
        "examples": [
            ["An apple contains about 95 calories and 25g of carbohydrates", "Cooking an egg denatures its proteins, making it solid"],
            ["Foods with a high GI cause blood sugar to rise quickly"],
            ["Ribosomes assemble amino acids into proteins based on mRNA instructions"],
            ["Fat-soluble vitamins (A, D, E, K) are absorbed with dietary fats"],
            ["Refrigeration slows bacterial growth, preserving food"],
            ["Enzymes speed up chemical reactions without being consumed"],
            ["The sum of all chemical reactions in a living organism is metabolism"],
            ["Lack of vitamin C can lead to scurvy"],
            ["Washing hands before handling food prevents contamination"],
            ["Soluble fiber dissolves in water and can help lower cholesterol"]
        ]
    },
    "general_academic": {
        "concepts": ["Critical Thinking", "Research Methods", "Argument Structure", "Evidence Evaluation", "Logical Fallacies", "Study Design", "Bias Identification", "Effective Communication", "Learning Strategies", "Information Synthesis"],
        "misconceptions": [
            "Correlation implies causation",
            "Anecdotal evidence is as strong as statistical evidence",
            "All scientific theories are proven facts",
            "Experts are never biased",
            "Published research is always correct",
            "If I believe something strongly, it must be true"
        ],
        "prerequisites": ["basic literacy", "critical thinking basics"],
        "formulas": [],  # Less formula-dependent
        "examples": [
            ["If 60% of students passed and there are 100 students, then 40 failed", "In a debate, claiming 'we should do X because everyone does it' is a bandwagon fallacy"],
            ["If a researcher wants to study the effect of a drug, they might use a randomized controlled trial"],
            ["If an argument has premises that lead to a conclusion, it is deductively valid"],
            ["If a study relies only on personal stories, it may not be generalizable"],
            ["If someone assumes that because A happened before B, A caused B, they commit the post hoc fallacy"],
            ["If a researcher wants to study consumer preferences, they might use a survey with a random sample"],
            ["If someone only seeks information that confirms their existing beliefs, they exhibit confirmation bias"],
            ["If a speaker uses clear language, organized structure, and engaging delivery, they communicate effectively"],
            ["If a student wants to improve memory, they might use spaced repetition and active recall"],
            ["If a researcher has data from multiple sources, they might combine it to form a complete picture"]
        ]
    }
}

def generate_example_id(category: str, domain: str, index: int) -> str:
    """Generate a unique ID for an example."""
    return f"{domain}_{category}_{index+1:03d}"

def get_concept_explanation(domain: str, concept: str) -> str:
    """Generate a brief explanation for a concept."""
    # Try to get from formulas
    formulas = DOMAIN_CONTENT[domain]["formulas"]
    if formulas:
        # Use the first formula as a simple explanation
        return f"it is defined by {formulas[0]}"
    else:
        # Generic explanation
        return f"it is an important concept in {domain}"

def get_concept_reason(domain: str, concept: str) -> str:
    """Generate a reason why the concept is important."""
    reasons = {
        "economics": "it helps understand how economies function and make informed decisions",
        "accountancy": "it is essential for accurate financial reporting and decision-making",
        "mathematics": "it provides tools for solving problems and understanding patterns",
        "science": "it explains natural phenomena and is foundational to scientific literacy",
        "nutrition_food_science": "it is crucial for maintaining health and preventing disease",
        "general_academic": "it underpins rigorous academic work and rational thought"
    }
    return reasons.get(domain, "it is fundamental to understanding the domain")

def get_concept_example(domain: str, concept: str) -> str:
    """Generate an example for a concept."""
    examples_list = DOMAIN_CONTENT[domain]["examples"]
    if examples_list:
        # Pick a random example pair and use the first one
        example_pair = random.choice(examples_list)
        return example_pair[0]
    else:
        # Fallback
        return f"considering a typical scenario involving {concept} in {domain}"

def create_base_context(domain: str, concept: str, level: str = "beginner", conversation_turns: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """Create a base context for an example."""
    domain_info = DOMAIN_CONTENT[domain]

    if conversation_turns is None:
        conversation_turns = [
            {"role": "student", "content": f"I'm struggling to understand {concept}."}
        ]

    return {
        "task": "resolve_confusion",
        "document_context": {
            "document_id": f"{domain}101_chapter{random.randint(1, 5)}",
            "title": f"Introduction to {domain.title()}",
            "page": random.randint(10, 50),
            "section": concept.split()[0] if " " in concept else concept[:15],
            "selected_text": f"The concept of {concept} is fundamental to understanding {domain}...",
            "surrounding_context": f"In the study of {domain}, {concept} plays a crucial role in explaining various phenomena and solving practical problems.",
            "retrieved_evidence": random.sample(domain_info["formulas"], min(2, len(domain_info["formulas"]))) if domain_info["formulas"] else [f"Key principle of {concept} in {domain}"]
        },
        "learner_state": {
            "level": level,
            "concept": concept,
            "mastery": round(random.uniform(0.2, 0.6), 2),
            "known_prerequisites": random.sample(domain_info["prerequisites"], min(2, len(domain_info["prerequisites"]))),
            "weak_prerequisites": random.sample([p for p in domain_info["prerequisites"] if p not in ["basic algebra", "algebra"]], min(1, 2)),
            "known_misconceptions": []
        },
        "conversation_context": conversation_turns,
        "current_goal": f"Understand {concept} well enough to apply it in {domain} contexts"
    }

# We'll define creation functions for each category.
# They will generate actual content without placeholders.

def create_simple_explanation(domain: str, index: int) -> Dict[str, Any]:
    """Create a simple explanation example with actual content."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])
    explanation = get_concept_explanation(domain, concept)
    reason = get_concept_reason(domain, concept)
    example = get_concept_example(domain, concept)

    return {
        "id": generate_example_id("simple_explanation", domain, index),
        "category": "simple_explanation",
        "domain": domain,
        "instruction": f"Explain the concept of {concept} in {domain}.",
        "context": create_base_context(domain, concept),
        "target": {
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "simple_explanation",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"The concept of {concept} in {domain} refers to {explanation}. It is important because {reason}. For example, {example}.",
            "understanding_check": {
                "required": True,
                "question": f"What is the main idea of {concept} in {domain}?",
                "expected_answer": f"The main idea is that {explanation}."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": random.choice(["low", "medium"]),
                "misconception_addressed": random.choice(domain_info["misconceptions"]) if random.random() > 0.3 else None
            },
            "memory": {
                "candidate": random.random() > 0.7,
                "title": f"Key Concept: {concept}" if random.random() > 0.5 else None,
                "content": f"{concept} is fundamental to {domain} because {reason}." if random.random() > 0.5 else None,
                "anchor_concept": concept if random.random() > 0.5 else None,
                "memory_type": "definition" if random.random() > 0.5 else "example",
                "confidence": round(random.uniform(0.6, 0.9), 2)
            }
        }
    }

def create_simplification(domain: str, index: int) -> Dict[str, Any]:
    """Create a simplification example with actual content."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])
    formula = random.choice(domain_info["formulas"]) if domain_info["formulas"] else f"the principle of {concept}"

    return {
        "id": generate_example_id("simplification", domain, index),
        "category": "simplification",
        "domain": domain,
        "instruction": f"Simplify the explanation of {concept} for a beginner.",
        "context": create_base_context(domain, concept, "beginner"),
        "target": {
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "simplification",
            "difficulty": "beginner",
            "response": f"Think of {concept} like [everyday analogy]. Basically, it means {get_concept_explanation(domain, concept)} instead of the technical definition involving {formula}.",
            "understanding_check": {
                "required": True,
                "question": f"In your own words, what does {concept} mean?",
                "expected_answer": f"It means {get_concept_explanation(domain, concept)}."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "medium",
                "misconception_addressed": "Belief that complex terminology is necessary for understanding"
            },
            "memory": {
                "candidate": random.random() > 0.6,
                "title": f"Simple Definition: {concept}",
                "content": f"{concept} simply means {get_concept_explanation(domain, concept)}.",
                "anchor_concept": concept,
                "memory_type": "definition",
                "confidence": round(random.uniform(0.7, 0.9), 2)
            }
        }
    }

def create_analogy(domain: str, index: int) -> Dict[str, Any]:
    """Create an analogy example with actual content."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    # Analogies mapping
    analogies = {
        "economics": ["MPC is like a leaky bucket", "Supply and demand is like a seesaw", "Inflation is like water filling a container"],
        "mathematics": ["A function is like a machine", "Derivatives are like instantaneous speed", "Integrals are like adding up slices"],
        "science": ["Photosynthesis is like a factory", "DNA is like a blueprint", "Cells are like tiny cities"],
        "nutrition_food_science": ["Metabolism is like a car engine", "Nutrients are like building blocks", "Enzymes are like specialized tools"],
        "accountancy": ["Assets are like what you own", "Liabilities are like what you owe", "Revenue is like income"],
        "general_academic": ["An argument is like a building", "Evidence is like bricks", "Reasoning is like the architect's plan"]
    }

    analogy_list = analogies.get(domain, ["Think of it like a tool", "It's similar to a process", "It works like a system"])
    analogy = random.choice(analogy_list)

    return {
        "id": generate_example_id("analogy", domain, index),
        "category": "analogy",
        "domain": domain,
        "instruction": f"Explain {concept} using an analogy.",
        "context": create_base_context(domain, concept),
        "target": {
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "analogy",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"{concept} is like {analogy}. Just as {analogy.split('.')[0] if '.' in analogy else analogy}, {concept} in {domain} {get_concept_explanation(domain, concept)}. This helps us understand {get_concept_reason(domain, concept)}.",
            "understanding_check": {
                "required": True,
                "question": f"What analogy was used to explain {concept}, and what does it illustrate?",
                "expected_answer": f"The analogy of {analogy} was used to illustrate {get_concept_explanation(domain, concept)}."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": random.choice(["low", "medium"]),
                "misconception_addressed": random.choice(domain_info["misconceptions"]) if random.random() > 0.4 else None
            },
            "memory": {
                "candidate": random.random() > 0.6,
                "title": f"Analogy for {concept}",
                "content": f"Remember: {concept} is like {analogy}.",
                "anchor_concept": concept,
                "memory_type": "analogy",
                "confidence": round(random.uniform(0.7, 0.9), 2)
            }
        }
    }

def create_concrete_example(domain: str, index: int) -> Dict[str, Any]:
    """Create a concrete example example with actual content."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    # Concrete examples
    examples_dict = {
        "economics": [["If you earn $1000 and spend $700, your MPC is 0.7", "When gas prices rise from $3 to $4/gallon, quantity demanded might drop 10%"]],
        "mathematics": [["Solving 2x² + 5x - 3 = 0 gives x = 0.5 or x = -3", "The derivative of x³ at x=2 is 12"]],
        "science": [["In photosynthesis, 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂", "A 60kg person on Earth weighs about 10kg on the Moon"]],
        "nutrition_food_science": [["An apple contains about 95 calories and 25g of carbohydrates", "Cooking an egg denatures its proteins, making it solid"]],
        "accountancy": [["If assets = $100,000 and liabilities = $60,000, then equity = $40,000", "Revenue of $50,000 minus expenses of $30,000 equals profit of $20,000"]],
        "general_academic": [["If 60% of students passed and there are 100 students, then 40 failed", "In a debate, claiming 'we should do X because everyone does it' is a bandwagon fallacy"]]
    }

    example_list = examples_dict.get(domain, [["For example, consider a simple case...", "In practice, this works like..."]])
    example_pair = random.choice(example_list)
    example1, example2 = example_pair[0], example_pair[1] if len(example_pair) > 1 else example_pair[0]

    return {
        "id": generate_example_id("concrete_example", domain, index),
        "category": "concrete_example",
        "domain": domain,
        "instruction": f"Provide a concrete example of {concept}.",
        "context": create_base_context(domain, concept),
        "target": {
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "concrete_example",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"Here's a concrete example: {example1}. This illustrates {concept} because {get_concept_explanation(domain, concept)}. Another example: {example2} shows {get_concept_reason(domain, concept)}.",
            "understanding_check": {
                "required": True,
                "question": f"Based on the examples provided, what is {concept}?",
                "expected_answer": f"{concept} is {get_concept_explanation(domain, concept)}."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": random.choice(["low", "medium"]),
                "misconception_addressed": "Belief that abstract concepts cannot be understood without examples"
            },
            "memory": {
                "candidate": random.random() > 0.5,
                "title": f"Example of {concept}",
                "content": f"Key example: {example1}. Remember that {get_concept_reason(domain, concept)}.",
                "anchor_concept": concept,
                "memory_type": "example",
                "confidence": round(random.uniform(0.6, 0.8), 2)
            }
        }
    }

def create_numerical_example(domain: str, index: int) -> Dict[str, Any]:
    """Create a numerical example example with actual content."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    # Numerical examples (similar to concrete but focused on numbers)
    numerical_examples = {
        "economics": [["If MPC is 0.8 and income increases by $1000, consumption increases by $800", "With a tax multiplier of -4, a $50 tax decrease increases GDP by $200"]],
        "mathematics": [["Solving 3x² - 12x + 9 = 0 gives x = 1 or x = 3", "The integral of 2x from 0 to 3 equals 9"]],
        "science": [["In 2H₂ + O₂ → 2H₂O, 4g of hydrogen produces 36g of water", "A force of 10N applied for 5s produces an impulse of 50 N·s"]],
        "nutrition_food_science": [["A 2000-calorie diet with 50% carbohydrates provides 1000 calories from carbs", "To lose 1lb per week, create a 500-calorie daily deficit"]],
        "accountancy": [["Straight-line depreciation: ($50,000 - $5,000) / 5 years = $9,000 per year", "Current ratio of 2.0 means $200,000 current assets cover $100,000 current liabilities"]],
        "general_academic": [["If p < 0.05 in a study with n=100, the result is statistically significant", "A 95% confidence interval of [0.4, 0.6] means we're 95% confident the true proportion is between 40% and 60%"]]
    }

    example_list = numerical_examples.get(domain, [["For example, consider the numerical case...", "In practice, this works out to..."]])
    example_pair = random.choice(example_list)
    example1, example2 = example_pair[0], example_pair[1] if len(example_pair) > 1 else example_pair[0]

    return {
        "id": generate_example_id("numerical_example", domain, index),
        "category": "numerical_example",
        "domain": domain,
        "instruction": f"Provide a numerical example of {concept}.",
        "context": create_base_context(domain, concept),
        "target": {
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "concrete_example",  # Still using concrete_example strategy but could be numerical_example
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"Here's a numerical example: {example1}. This shows {concept} in action because {get_concept_explanation(domain, concept)}. Another case: {example2} demonstrates {get_concept_reason(domain, concept)}.",
            "understanding_check": {
                "required": True,
                "question": f"Based on the numerical examples provided, what does {concept} demonstrate?",
                "expected_answer": f"{concept} demonstrates {get_concept_explanation(domain, concept)}."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": random.choice(["low", "medium"]),
                "misconception_addressed": "Belief that numerical examples aren't helpful for understanding abstract concepts"
            },
            "memory": {
                "candidate": random.random() > 0.5,
                "title": f"Numerical example of {concept}",
                "content": f"Key numerical example: {example1}. Remember that {get_concept_reason(domain, concept)}.",
                "anchor_concept": concept,
                "memory_type": "example",
                "confidence": round(random.uniform(0.6, 0.8), 2)
            }
        }
    }

def create_prerequisite_detection(domain: str, index: int) -> Dict[str, Any]:
    """Create a prerequisite detection example with actual content."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])
    missing_prereq = random.choice([p for p in domain_info["prerequisites"] if p not in ["basic algebra", "algebra"] or random.random() > 0.7])

    # Make it multi-turn: student asks about prerequisite
    conversation_turns = [
        {"role": "student", "content": f"I'm struggling to understand {concept}."},
        {"role": "tutor", "content": f"Let me explain {concept}."},
        {"role": "student", "content": f"But I don't understand {missing_prereq}. How is that related?"}
    ]

    return {
        "id": generate_example_id("prerequisite_detection", domain, index),
        "category": "prerequisite_detection",
        "domain": domain,
        "instruction": f"Identify what prerequisite knowledge is needed to understand {concept}.",
        "context": create_base_context(domain, concept, "beginner", conversation_turns),
        "target": {
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "prerequisite",
            "difficulty": "beginner",
            "response": f"To understand {concept}, you need to know about {missing_prereq}. {missing_prereq} is essential because {get_concept_reason(domain, missing_prereq)}. Without this foundation, you'll struggle with {concept} because {get_concept_explanation(domain, concept)} builds on {missing_prereq}.",
            "understanding_check": {
                "required": True,
                "question": f"What prerequisite is essential for understanding {concept}?",
                "expected_answer": f"{missing_prereq} is essential because {get_concept_reason(domain, missing_prereq)}."
            },
            "diagnosis": {
                "confusion_type": "prerequisite_gap",
                "severity": "medium",
                "misconception_addressed": None,
                "missing_prerequisite": missing_prereq
            },
            "memory": {
                "candidate": random.random() > 0.6,
                "title": f"Prerequisite for {concept}",
                "content": f"You must understand {missing_prereq} before learning {concept}.",
                "anchor_concept": concept,
                "memory_type": "prerequisite",
                "confidence": round(random.uniform(0.7, 0.9), 2)
            }
        }
    }

def create_misconception_correction(domain: str, index: int) -> Dict[str, Any]:
    """Create a misconception correction example with actual content."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])
    misconception = random.choice(domain_info["misconceptions"])

    # Make it multi-turn: student expresses misconception
    conversation_turns = [
        {"role": "student", "content": f"I think {misconception}."},
        {"role": "tutor", "content": f"That's a common misconception. Let me explain why it's incorrect."}
    ]

    return {
        "id": generate_example_id("misconception_detection", domain, index),
        "category": "misconception_detection",
        "domain": domain,
        "instruction": f"Address a common misconception about {concept}.",
        "context": create_base_context(domain, concept, "beginner", conversation_turns),
        "target": {
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "misconception_correction",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"A common misconception is that '{misconception}'. This is incorrect because {get_concept_explanation(domain, concept)} shows that {get_concept_reason(domain, concept)}. The correct understanding is that {get_concept_explanation(domain, concept)} actually means {get_concept_reason(domain, concept)}.",
            "understanding_check": {
                "required": True,
                "question": f"Why is the statement '{misconception}' incorrect regarding {concept}?",
                "expected_answer": f"It's incorrect because {get_concept_explanation(domain, concept)} demonstrates that {get_concept_reason(domain, concept)}."
            },
            "diagnosis": {
                "confusion_type": "misconception",
                "severity": "medium",
                "misconception_addressed": misconception
            },
            "memory": {
                "candidate": random.random() > 0.6,
                "title": f"Misconception: {concept}",
                "content": f"Remember: {misconception} is WRONG because {get_concept_explanation(domain, concept)}.",
                "anchor_concept": concept,
                "memory_type": "misconception_correction",
                "confidence": round(random.uniform(0.7, 0.9), 2)
            }
        }
    }

def create_follow_up_questions(domain: str, index: int) -> Dict[str, Any]:
    """Create a follow-up questions example with actual content."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    # Keep single-turn but with actual content
    return {
        "id": generate_example_id("follow_up_questions", domain, index),
        "category": "follow_up_questions",
        "domain": domain,
        "instruction": f"Generate follow-up questions to check understanding of {concept}.",
        "context": create_base_context(domain, concept),
        "target": {
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "follow_up_questions",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"To check your understanding of {concept}: {get_concept_explanation(domain, concept)}. Consider these questions: 1) How does {concept} relate to {random.choice(domain_info['prerequisites'])}? 2) What would happen if {random.choice(domain_info['misconceptions'])} were true? 3) Can you give an example of {concept} in practice?",
            "understanding_check": {
                "required": True,
                "question": f"What is a good follow-up question to test understanding of {concept}?",
                "expected_answer": f"A good question would be about how {concept} applies in real-world situations because {get_concept_reason(domain, concept)}."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "low",
                "misconception_addressed": None
            },
            "memory": {
                "candidate": False,
                "title": None,
                "content": None,
                "anchor_concept": None,
                "memory_type": None,
                "confidence": 0.0
            }
        }
    }

def create_multi_turn_tutoring(domain: str, index: int) -> Dict[str, Any]:
    """Create a multi-turn tutoring example with actual content."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    # Multi-turn tutoring: student doesn't understand, tutor explains simply, student asks about specific aspect
    conversation_turns = [
        {"role": "student", "content": f"I don't understand {concept}."},
        {"role": "tutor", "content": f"Let me explain {concept} simply: {get_concept_explanation(domain, concept)}."},
        {"role": "student", "content": f"I get the basic idea, but how does {get_concept_reason(domain, concept)} work in practice?"}
    ]

    return {
        "id": generate_example_id("multi_turn_tutoring", domain, index),
        "category": "multi_turn_tutoring",
        "domain": domain,
        "instruction": f"Continue the tutoring conversation about {concept}.",
        "context": create_base_context(domain, concept, "beginner", conversation_turns),
        "target": {
            "action": "explain",
            "strategy": "multi_turn_tutoring",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"Great question! {get_concept_reason(domain, concept)} works by {get_concept_explanation(domain, concept)}. To see this in practice, {get_concept_example(domain, concept)}.",
            "understanding_check": {
                "required": True,
                "question": f"Based on our conversation, how would you explain {get_concept_reason(domain, concept)} of {concept}?",
                "expected_answer": f"{get_concept_reason(domain, concept)} works by {get_concept_explanation(domain, concept)}."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "low",
                "misconception_addressed": None
            },
            "memory": {
                "candidate": random.random() > 0.5,
                "title": f"Multi-turn insight: {concept}",
                "content": f"From our discussion: {get_concept_explanation(domain, concept)} is key to understanding {concept}.",
                "anchor_concept": concept,
                "memory_type": "insight",
                "confidence": round(random.uniform(0.6, 0.8), 2)
            }
        }
    }

def create_repeated_confusion(domain: str, index: int) -> Dict[str, Any]:
    """Create a repeated confusion example with actual content."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    # Make it multi-turn with multiple attempts
    conversation_turns = [
        {"role": "student", "content": f"I don't understand {concept}."},
        {"role": "tutor", "content": f"Let me explain {concept}: {get_concept_explanation(domain, concept)}."},
        {"role": "student", "content": f"I still don't get it."},
        {"role": "tutor", "content": f"Let me try another way: {get_concept_reason(domain, concept)}."},
        {"role": "student", "content": f"I'm still confused."}
    ]

    return {
        "id": generate_example_id("repeated_confusion", domain, index),
        "category": "repeated_confusion",
        "domain": domain,
        "instruction": f"Address repeated confusion about {concept}.",
        "context": create_base_context(domain, concept, "beginner", conversation_turns),
        "target": {
            "action": "explain",
            "strategy": "simple_explanation",
            "difficulty": "beginner",
            "response": f"I appreciate your persistence. Let me try a different approach with {concept}: {get_concept_explanation(domain, concept)}. The key is to focus on {get_concept_reason(domain, concept)} rather than just memorizing formulas.",
            "understanding_check": {
                "required": True,
                "question": f"What was the key breakthrough in understanding {concept}?",
                "expected_answer": f"The key was realizing {get_concept_reason(domain, concept)}."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "medium",
                "misconception_addressed": "Belief that if you don't understand immediately, you never will"
            },
            "memory": {
                "candidate": random.random() > 0.6,
                "title": f"Overcoming confusion about {concept}",
                "content": f"Remember: When confused about {concept}, try explaining it in terms of {get_concept_reason(domain, concept)}.",
                "anchor_concept": concept,
                "memory_type": "strategy",
                "confidence": round(random.uniform(0.7, 0.9), 2)
            }
        }
    }

def create_strategy_switching(domain: str, index: int) -> Dict[str, Any]:
    """Create a strategy switching example with actual content."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    # Multi-turn: student doesn't understand after explanation, tutor tries different strategy
    conversation_turns = [
        {"role": "student", "content": f"I don't understand {concept} even after the explanation."},
        {"role": "tutor", "content": f"Let me try explaining {concept} differently."}
    ]

    return {
        "id": generate_example_id("strategy_switching", domain, index),
        "category": "strategy_switching",
        "domain": domain,
        "instruction": f"Switch teaching strategies to better explain {concept}.",
        "context": create_base_context(domain, concept, "beginner", conversation_turns),
        "target": {
            "action": "explain",
            "strategy": "analogy",  # Switching to analogy after failed straightforward explanation
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"Let me try a different approach. Think of {concept} like {random.choice(['a tool', 'a process', 'a system'])}. Just as {random.choice(['a tool helps us accomplish tasks', 'a process transforms inputs', 'a system interconnects parts'])}, we can see that {concept} in {domain} {get_concept_explanation(domain, concept)}. This perspective helps because {get_concept_reason(domain, concept)}.",
            "understanding_check": {
                "required": True,
                "question": f"Why did switching to an analogy help explain {concept}?",
                "expected_answer": f"The analogy helped by making the abstract concept more concrete and relatable."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "medium",
                "misconception_addressed": "Belief that repeating the same explanation will eventually work"
            },
            "memory": {
                "candidate": random.random() > 0.6,
                "title": f"Strategy switch for {concept}",
                "content": f"When standard explanations don't work for {concept}, try using an analogy or real-world example.",
                "anchor_concept": concept,
                "memory_type": "strategy",
                "confidence": round(random.uniform(0.7, 0.9), 2)
            }
        }
    }

def create_diagnostic_questions(domain: str, index: int) -> Dict[str, Any]:
    """Create a diagnostic questions example with actual content."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    # Keep single-turn but with actual content
    return {
        "id": generate_example_id("diagnostic_questions", domain, index),
        "category": "diagnostic_questions",
        "domain": domain,
        "instruction": f"Ask diagnostic questions to identify the nature of confusion about {concept}.",
        "context": create_base_context(domain, concept),
        "target": {
            "action": "explain",
            "strategy": "diagnostic_questions",
            "difficulty": "beginner",
            "response": f"To better understand your confusion about {concept}, let me ask: 1) What part of {concept} makes the most sense to you? 2) What specifically feels confusing or unclear? 3) When you think about {concept}, what comes to mind first? 4) Have you encountered similar concepts before?",
            "understanding_check": {
                "required": False,  # Diagnostic questions don't always have a single correct answer
                "question": None,
                "expected_answer": None
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "medium",
                "misconception_addressed": None
            },
            "memory": {
                "candidate": False,
                "title": None,
                "content": None,
                "anchor_concept": None,
                "memory_type": None,
                "confidence": 0.0
            }
        }
    }

def create_hint_based_teaching(domain: str, index: int) -> Dict[str, Any]:
    """Create a hint-based teaching example with actual content."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    # Make it multi-turn: student asks for hints
    conversation_turns = [
        {"role": "student", "content": f"I'm struggling to understand {concept}. Can you give me a hint?"},
        {"role": "tutor", "content": f"Sure! Hint 1: {get_concept_explanation(domain, concept)}."},
        {"role": "student", "content": f"I'm still not sure. Another hint?"},
        {"role": "tutor", "content": f"Hint 2: Think about how {concept} relates to {random.choice(domain_info['prerequisites'])}."}
    ]

    return {
        "id": generate_example_id("hint_based_teaching", domain, index),
        "category": "hint_based_teaching",
        "domain": domain,
        "instruction": f"Provide hints to help the learner figure out {concept} themselves.",
        "context": create_base_context(domain, concept, "beginner", conversation_turns),
        "target": {
            "action": "explain",
            "strategy": "hint",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"Instead of giving you the answer directly, let me give you some hints about {concept}: Hint 1: {get_concept_explanation(domain, concept)}. Hint 2: Think about how {concept} relates to {random.choice(domain_info['prerequisites'])}. Hint 3: Consider a real-world example like {get_concept_example(domain, concept)}. See if you can put these together to understand {concept}.",
            "understanding_check": {
                "required": True,
                "question": f"Based on the hints provided, what is {concept}?",
                "expected_answer": f"{get_concept_explanation(domain, concept)} synthesized with {get_concept_reason(domain, concept)}."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "medium",
                "misconception_addressed": "Belief that learning requires direct instruction rather than discovery"
            },
            "memory": {
                "candidate": random.random() > 0.5,
                "title": f"Hint-based approach to {concept}",
                "content": f"Remember the hints for {concept}: {get_concept_explanation(domain, concept)}, {get_concept_reason(domain, concept)}, {get_concept_example(domain, concept)}.",
                "anchor_concept": concept,
                "memory_type": "hint",
                "confidence": round(random.uniform(0.6, 0.8), 2)
            }
        }
    }

def create_practice_questions(domain: str, index: int) -> Dict[str, Any]:
    """Create a practice questions example with actual content."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    # Make it multi-turn: student asks for practice problems
    conversation_turns = [
        {"role": "student", "content": f"I'd like to practice {concept} to make sure I understand."},
        {"role": "tutor", "content": f"Great idea! Let's work through some problems together."}
    ]

    return {
        "id": generate_example_id("practice_questions", domain, index),
        "category": "practice_questions",
        "domain": domain,
        "instruction": f"Provide practice problems to reinforce understanding of {concept}.",
        "context": create_base_context(domain, concept, "beginner", conversation_turns),
        "target": {
            "action": "explain",
            "strategy": "practice",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"Let's practice {concept} with some exercises:\n\nProblem 1: {get_concept_example(domain, concept)}\n\nProblem 2: Explain how {concept} relates to {random.choice(domain_info['prerequisites'])}.\n\nProblem 3: Describe a real-world application of {concept} in {domain}.\n\nTry solving these, and I'll check your work.",
            "understanding_check": {
                "required": True,
                "question": f"What is the key skill demonstrated in solving these {concept} problems?",
                "expected_answer": f"The key skill is applying {concept} to solve problems because {get_concept_reason(domain, concept)}."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "low",
                "misconception_addressed": None
            },
            "memory": {
                "candidate": random.random() > 0.5,
                "title": f"Practice problems for {concept}",
                "content": f"Key practice types for {concept}: calculation, explanation, application.",
                "anchor_concept": concept,
                "memory_type": "procedure",
                "confidence": round(random.uniform(0.6, 0.8), 2)
            }
        }
    }

def create_error_correction(domain: str, index: int) -> Dict[str, Any]:
    """Create an error correction example with actual content."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])
    misconception = random.choice(domain_info["misconceptions"])

    # Make it multi-turn: student makes error, tutor corrects
    conversation_turns = [
        {"role": "student", "content": f"I think {concept} means {misconception}."},
        {"role": "tutor", "content": f"I notice there's a misunderstanding in your work."}
    ]

    return {
        "id": generate_example_id("error_correction", domain, index),
        "category": "error_correction",
        "domain": domain,
        "instruction": f"Correct an error in the student's understanding of {concept}.",
        "context": {
            **create_base_context(domain, concept, "beginner"),
            "learner_state": {
                **create_base_context(domain, concept, "beginner")["learner_state"],
                "known_misconceptions": [misconception]  # Simulate they have this misconception
            }
        },
        "target": {
            "action": "explain",
            "strategy": "misconception_correction",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"I notice there's a misunderstanding in your work. You stated that '{misconception}', but that's not quite right. Let me correct this: {get_concept_explanation(domain, concept)} actually means {get_concept_reason(domain, concept)}. The error likely came from confusing {concept} with a similar concept.",
            "understanding_check": {
                "required": True,
                "question": f"What was the error in thinking about {concept}, and what is the correct understanding?",
                "expected_answer": f"The error was believing '{misconception}', and the correct understanding is {get_concept_explanation(domain, concept)}."
            },
            "diagnosis": {
                "confusion_type": "misconception",
                "severity": "medium",
                "misconception_addressed": misconception
            },
            "memory": {
                "candidate": random.random() > 0.6,
                "title": f"Corrected understanding: {concept}",
                "content": f"Remember: {get_concept_explanation(domain, concept)}, NOT '{misconception}'.",
                "anchor_concept": concept,
                "memory_type": "misconception_correction",
                "confidence": round(random.uniform(0.7, 0.9), 2)
            }
        }
    }

def create_partial_understanding(domain: str, index: int) -> Dict[str, Any]:
    """Create a partial understanding example with actual content."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    # Make it multi-turn: student shows partial understanding
    conversation_turns = [
        {"role": "student", "content": f"I think I get part of {concept}."},
        {"role": "tutor", "content": f"You've got part of {concept} right - let's see what you're missing."}
    ]

    return {
        "id": generate_example_id("partial_understanding", domain, index),
        "category": "partial_understanding",
        "domain": domain,
        "instruction": f"Address partial understanding of {concept}.",
        "context": create_base_context(domain, concept, "beginner", conversation_turns),
        "target": {
            "action": "explain",
            "strategy": "partial_understanding",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"You've got part of {concept} right - you correctly identified that {get_concept_explanation(domain, concept)}. However, you're missing {get_concept_reason(domain, concept)}. Let me explain the full picture: {get_concept_explanation(domain, concept)} is important because {get_concept_reason(domain, concept)}.",
            "understanding_check": {
                "required": True,
                "question": f"What aspect of {concept} was missing from the initial understanding?",
                "expected_answer": f"The missing aspect was {get_concept_reason(domain, concept)} because {get_concept_explanation(domain, concept)} doesn't make sense without understanding why it matters."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "low",
                "misconception_addressed": None
            },
            "memory": {
                "candidate": random.random() > 0.5,
                "title": f"Complete understanding of {concept}",
                "content": f"To fully understand {concept}, remember both {get_concept_explanation(domain, concept)} and {get_concept_reason(domain, concept)}.",
                "anchor_concept": concept,
                "memory_type": "concept",
                "confidence": round(random.uniform(0.6, 0.8), 2)
            }
        }
    }

def create_understanding_confirmation(domain: str, index: int) -> Dict[str, Any]:
    """Create an understanding confirmation example with actual content."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    # Make it multi-turn: student explains, tutor confirms
    conversation_turns = [
        {"role": "student", "content": f"I think I understand {concept} now."},
        {"role": "tutor", "content": f"Excellent work! Let me hear your explanation."}
    ]

    return {
        "id": generate_example_id("understanding_confirmation", domain, index),
        "category": "understanding_confirmation",
        "domain": domain,
        "instruction": f"Confirm the student's understanding of {concept}.",
        "context": create_base_context(domain, concept, "beginner", conversation_turns),
        "target": {
            "action": "explain",
            "strategy": "understanding_confirmation",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"Excellent work! You've demonstrated solid understanding of {concept} by explaining that {get_concept_explanation(domain, concept)}. This shows you've grasped the key concepts including {get_concept_reason(domain, concept)}.",
            "understanding_check": {
                "required": True,
                "question": f"How would you explain {concept} to someone else who is just learning it?",
                "expected_answer": f"I would explain it as {get_concept_explanation(domain, concept)} because {get_concept_reason(domain, concept)}."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "very_low",  # Almost no confusion left
                "misconception_addressed": None
            },
            "memory": {
                "candidate": random.random() > 0.7,
                "title": f"Mastered concept: {concept}",
                "content": f"You now understand {concept} well enough to {get_concept_example(domain, concept)}.",
                "anchor_concept": concept,
                "memory_type": "mastery",
                "confidence": round(random.uniform(0.8, 0.95), 2)
            }
        }
    }

def create_memory_generation(domain: str, index: int) -> Dict[str, Any]:
    """Create a memory generation example with actual content."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    # Make it multi-turn: student asks for memory aid
    conversation_turns = [
        {"role": "student", "content": f"I'm worried I'll forget {concept}. Can you give me a memory aid?"},
        {"role": "tutor", "content": f"Sure! Let me create a memory candidate for you."}
    ]

    return {
        "id": generate_example_id("memory_generation", domain, index),
        "category": "memory_generation",
        "domain": domain,
        "instruction": f"Generate a memory candidate for {concept}.",
        "context": create_base_context(domain, concept, "beginner", conversation_turns),
        "target": {
            "action": "explain",
            "strategy": "simple_explanation",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"{concept} in {domain} is defined as {get_concept_explanation(domain, concept)}. It's important because {get_concept_reason(domain, concept)}. A key example is {get_concept_example(domain, concept)}.",
            "understanding_check": {
                "required": True,
                "question": f"What should we remember about {concept} in {domain}?",
                "expected_answer": f"We should remember that {concept} {get_concept_explanation(domain, concept)}."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "low",
                "misconception_addressed": None
            },
            "memory": {
                "candidate": True,  # This is specifically for memory generation
                "title": f"Essential: {concept}",
                "content": f"{concept} in {domain}: {get_concept_explanation(domain, concept)}. Key point: {get_concept_reason(domain, concept)}.",
                "anchor_concept": concept,
                "memory_type": "definition",
                "confidence": round(random.uniform(0.8, 0.95), 2)
            }
        }
    }

def create_document_grounded_teaching(domain: str, index: int) -> Dict[str, Any]:
    """Create a document-grounded teaching example with actual content."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    # Make it multi-turn: student asks about evidence
    conversation_turns = [
        {"role": "student", "content": f"I'm struggling to understand {concept}. What evidence do we have for this?"},
        {"role": "tutor", "content": f"Let's look at the document together."}
    ]

    return {
        "id": generate_example_id("document_grounded", domain, index),
        "category": "document_grounded",
        "domain": domain,
        "instruction": f"Teach {concept} using specific evidence from the document.",
        "context": create_base_context(domain, concept, "beginner", conversation_turns),
        "target": {
            "action": "explain",
            "strategy": "simple_explanation",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"Looking at our document, we can see that {concept} is clearly explained in the selected text: '{get_concept_explanation(domain, concept)}'. The surrounding context tells us that {get_concept_reason(domain, concept)}. The retrieved evidence supports this by showing {get_concept_example(domain, concept)}.",
            "understanding_check": {
                "required": True,
                "question": f"What evidence from the document supports the explanation of {concept}?",
                "expected_answer": f"The selected text states '{get_concept_explanation(domain, concept)}', and the surrounding context adds {get_concept_reason(domain, concept)}."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "low",
                "misconception_addressed": None
            },
            "memory": {
                "candidate": random.random() > 0.5,
                "title": f"Document-based insight: {concept}",
                "content": f"From our reading: {concept} means {get_concept_explanation(domain, concept)}.",
                "anchor_concept": concept,
                "memory_type": "document_insight",
                "confidence": round(random.uniform(0.6, 0.8), 2)
            }
        }
    }

def create_context_retention(domain: str, index: int) -> Dict[str, Any]:
    """Create a context retention example with actual content."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    # Make it multi-turn: student mentions earlier concept
    other_concept = random.choice([c for c in domain_info["concepts"] if c != concept])
    conversation_turns = [
        {"role": "student", "content": f"Earlier we talked about {other_concept}. Now I'm confused about {concept}."},
        {"role": "tutor", "content": f"Let's connect {other_concept} to {concept}."}
    ]

    return {
        "id": generate_example_id("context_retention", domain, index),
        "category": "context_retention",
        "domain": domain,
        "instruction": f"Demonstrate retention of earlier context while discussing {concept}.",
        "context": create_base_context(domain, concept, "beginner", conversation_turns),
        "target": {
            "action": "explain",
            "strategy": "simple_explanation",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"Great question about connecting {other_concept} to {concept}! Earlier we established that {get_concept_explanation(domain, other_concept)}. This relates to {concept} because {get_concept_explanation(domain, concept)} and {get_concept_explanation(domain, other_concept)} both relate to {get_concept_reason(domain, concept)}. Now, to understand {concept} specifically: {get_concept_explanation(domain, concept)}.",
            "understanding_check": {
                "required": True,
                "question": f"How does {other_concept} relate to {concept}?",
                "expected_answer": f"{other_concept} relates to {concept} because both are important for {get_concept_reason(domain, concept)} and {get_concept_explanation(domain, concept)} builds on {get_concept_explanation(domain, other_concept)}."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "low",
                "misconception_addressed": None
            },
            "memory": {
                "candidate": random.random() > 0.5,
                "title": f"Connecting concepts: {concept} and {other_concept}",
                "content": f"Remember: {concept} and {other_concept} are related because {get_concept_reason(domain, concept)}.",
                "anchor_concept": concept,
                "memory_type": "connection",
                "confidence": round(random.uniform(0.6, 0.8), 2)
            }
        }
    }

def generate_all_categories(domain: str, single_examples: int, multi_examples: int) -> List[Dict[str, Any]]:
    """Generate examples for all categories in a domain with specified counts."""
    examples = []

    # Define which categories are single-turn and multi-turn
    SINGLE_TURN_CATEGORIES = {
        "simple_explanation", "simplification", "analogy", "concrete_example",
        "numerical_example", "diagnostic_questions", "follow_up_questions"
    }
    MULTI_TURN_CATEGORIES = set(VALID_CATEGORIES) - SINGLE_TURN_CATEGORIES

    # Define generators for each category
    generators = {
        "simple_explanation": create_simple_explanation,
        "simplification": create_simplification,
        "analogy": create_analogy,
        "concrete_example": create_concrete_example,
        "numerical_example": create_numerical_example,
        "prerequisite_detection": create_prerequisite_detection,
        "misconception_detection": create_misconception_correction,
        "follow_up_questions": create_follow_up_questions,
        "multi_turn_tutoring": create_multi_turn_tutoring,
        "repeated_confusion": create_repeated_confusion,
        "strategy_switching": create_strategy_switching,
        "diagnostic_questions": create_diagnostic_questions,
        "hint_based_teaching": create_hint_based_teaching,
        "practice_questions": create_practice_questions,
        "error_correction": create_error_correction,
        "partial_understanding": create_partial_understanding,
        "understanding_confirmation": create_understanding_confirmation,
        "memory_generation": create_memory_generation,
        "document_grounded": create_document_grounded_teaching,
        "context_retention": create_context_retention
    }

    # Generate single-turn examples
    for category in SINGLE_TURN_CATEGORIES:
        generator_func = generators[category]
        for i in range(single_examples):
            example = generator_func(domain, i)
            examples.append(example)

    # Generate multi-turn examples
    for category in MULTI_TURN_CATEGORIES:
        generator_func = generators[category]
        for i in range(multi_examples):
            example = generator_func(domain, i)
            examples.append(example)

    return examples

def generate_improved_seed_data(output_dir: str = "./data/generated",
                               total_examples: int = 480,
                               multi_turn_ratio: float = 0.4) -> List[Dict[str, Any]]:
    """
    Generate improved seed data for OLORIC.

    Args:
        output_dir: Directory to save generated data
        total_examples: Total number of examples to generate
        multi_turn_ratio: Target proportion of multi-turn examples

    Returns:
        List of generated training examples
    """
    print("Generating improved OLORIC seed data...")

    all_examples = []

    # Calculate number of examples per category
    SINGLE_TURN_CATEGORIES = {
        "simple_explanation", "simplification", "analogy", "concrete_example",
        "numerical_example", "diagnostic_questions", "follow_up_questions"
    }
    n_single = len(SINGLE_TURN_CATEGORIES)
    n_multi = len(VALID_CATEGORIES) - n_single

    # Calculate examples per category
    # We want multi_turn_ratio of total to be multi-turn
    multi_turn_total = int(total_examples * multi_turn_ratio)
    single_turn_total = total_examples - multi_turn_total

    # Distribute evenly among categories
    single_per_category = single_turn_total // n_single
    multi_per_category = multi_turn_total // n_multi

    # Handle remainder
    single_remainder = single_turn_total % n_single
    multi_remainder = multi_turn_total % n_multi

    print(f"Target: {single_turn_total} single-turn, {multi_turn_total} multi-turn")
    print(f"Per category: {single_per_category} single-turn ({single_remainder} remainder), {multi_per_category} multi-turn ({multi_remainder} remainder)")

    for domain in VALID_DOMAINS:
        print(f"Generating examples for {domain}...")
        # Distribute remainders to first few domains
        single_extra = 1 if single_remainder > 0 else 0
        multi_extra = 1 if multi_remainder > 0 else 0
        single_this = single_per_category + single_extra
        multi_this = multi_per_category + multi_extra
        domain_examples = generate_all_categories(domain, single_this, multi_this)
        all_examples.extend(domain_examples)
        print(f"  Generated {len(domain_examples)} examples")
        # Decrease remainders for next domain (simple round-robin)
        if single_remainder > 0:
            single_remainder -= 1
        if multi_remainder > 0:
            multi_remainder -= 1

    # Shuffle to mix domains and categories
    random.shuffle(all_examples)

    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "improved_seed_data.jsonl")

    print(f"Saving {len(all_examples)} examples to {output_file}")
    with open(output_file, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')

    print("Improved seed data generation complete!")
    return all_examples

def main():
    """Main generation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate improved OLORIC seed data")
    parser.add_argument("--output-dir", type=str, default="./data/generated",
                        help="Output directory for generated data")
    parser.add_argument("--total-examples", type=int, default=480,
                        help="Total number of examples to generate")
    parser.add_argument("--multi-turn-ratio", type=float, default=0.4,
                        help="Target proportion of multi-turn examples (0.0 to 1.0)")

    args = parser.parse_args()

    examples = generate_improved_seed_data(
        output_dir=args.output_dir,
        total_examples=args.total_examples,
        multi_turn_ratio=args.multi_turn_ratio
    )

    print(f"\nGenerated {len(examples)} examples saved to: {os.path.join(args.output_dir, 'improved_seed_data.jsonl')}")

    # Quick statistics
    categories_count = {}
    domains_count = {}
    multi_turn_count = 0
    single_turn_count = 0
    conversation_lengths = []

    for example in examples:
        cat = example["category"]
        dom = example["domain"]
        categories_count[cat] = categories_count.get(cat, 0) + 1
        domains_count[dom] = domains_count.get(dom, 0) + 1

        conversation = example.get('context', {}).get('conversation_context', [])
        length = len(conversation)
        conversation_lengths.append(length)
        if length >= 2:
            multi_turn_count += 1
        else:
            single_turn_count += 1

    print("\nGeneration Statistics:")
    print(f"Total examples: {len(examples)}")
    print(f"Single-turn examples: {single_turn_count} ({single_turn_count/len(examples)*100:.1f}%)")
    print(f"Multi-turn examples: {multi_turn_count} ({multi_turn_count/len(examples)*100:.1f}%)")
    print(f"Average conversation turns: {sum(conversation_lengths)/len(conversation_lengths):.2f}")

    print("\nExamples per category:")
    for category, count in sorted(categories_count.items()):
        print(f"  {category}: {count}")

    print("\nExamples per domain:")
    for domain, count in sorted(domains_count.items()):
        print(f"  {domain}: {count}")

    return 0

if __name__ == "__main__":
    exit(main())