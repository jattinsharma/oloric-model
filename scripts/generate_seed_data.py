#!/usr/bin/env python3
"""
Enhanced seed data generation script for OLORIC.
Generates initial high-quality training examples across all required categories and domains.
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
        "formulas": ["MPC = ΔC/ΔY", "GDP = C + I + G + (X-M)", "Elasticity = %ΔQ/%ΔP", "Multiplier = 1/(1-MPC)"]
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
        "formulas": ["quadratic formula: x = (-b ± √(b²-4ac))/2a", "derivative of xⁿ = nxⁿ⁻¹", "Pythagorean theorem: a²+b²=c²", "area of circle = πr²"]
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
        "formulas": ["F = ma", "E = mc²", "PV = nRT", "E = hf", "pH = -log[H⁺]"]
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
        "formulas": ["BMI = weight(kg)/height(m)²", "BMR = 10×weight(kg) + 6.25×height(cm) - 5×age(yr) + s", "Energy intake = Energy expenditure + Δbody stores"]
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
        "formulas": ["Assets = Liabilities + Equity", "Current Ratio = Current Assets/Current Liabilities", "Debt-to-Equity = Total Debt/Total Equity", "ROE = Net Income/Shareholder Equity"]
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
        "formulas": []  # Less formula-dependent
    }
}

def generate_example_id(category: str, domain: str, index: int) -> str:
    """Generate a unique ID for an example."""
    return f"{domain}_{category}_{index+1:03d}"

def create_base_context(domain: str, concept: str, level: str = "beginner") -> Dict[str, Any]:
    """Create a base context for an example."""
    domain_info = DOMAIN_CONTENT[domain]

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
        "conversation_context": [
            {"role": "student", "content": f"I'm struggling to understand {concept}."}
        ],
        "current_goal": f"Understand {concept} well enough to apply it in {domain} contexts"
    }

def create_simple_explanation(domain: str, index: int) -> Dict[str, Any]:
    """Create a simple explanation example."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    return {
        "id": generate_example_id("simple_explanation", domain, index),
        "category": "simple_explanation",
        "domain": domain,
        "instruction": f"Explain the concept of {concept} in {domain}.",
        "context": create_base_context(domain, concept),
        "target": {
            "action": "explain",
            "strategy": "simple_explanation",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"{concept} in {domain} refers to [detailed explanation]. It is important because [reason]. For example, [concrete example].",
            "understanding_check": {
                "required": True,
                "question": f"What is the main idea of {concept} in {domain}?",
                "expected_answer": f"The main idea is that [key point about concept]."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": random.choice(["low", "medium"]),
                "misconception_addressed": random.choice(domain_info["misconceptions"]) if random.random() > 0.3 else None
            },
            "memory": {
                "candidate": random.random() > 0.7,
                "title": f"Key Concept: {concept}" if random.random() > 0.5 else None,
                "content": f"{concept} is fundamental to {domain} because [reason]." if random.random() > 0.5 else None,
                "anchor_concept": concept if random.random() > 0.5 else None,
                "memory_type": "definition" if random.random() > 0.5 else "example",
                "confidence": round(random.uniform(0.6, 0.9), 2)
            }
        }
    }

def create_simplification(domain: str, index: int) -> Dict[str, Any]:
    """Create a simplification example."""
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
            "action": "explain",
            "strategy": "simplification",
            "difficulty": "beginner",
            "response": f"Think of {concept} like [everyday analogy]. Basically, it means [simple explanation] instead of the technical definition involving [formula/technical detail].",
            "understanding_check": {
                "required": True,
                "question": f"In your own words, what does {concept} mean?",
                "expected_answer": f"It means [simple, correct explanation]."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "medium",
                "misconception_addressed": "Belief that complex terminology is necessary for understanding"
            },
            "memory": {
                "candidate": random.random() > 0.6,
                "title": f"Simple Definition: {concept}",
                "content": f"{concept} simply means [clear, simple explanation].",
                "anchor_concept": concept,
                "memory_type": "definition",
                "confidence": round(random.uniform(0.7, 0.9), 2)
            }
        }
    }

def create_analogy(domain: str, index: int) -> Dict[str, Any]:
    """Create an analogy example."""
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
            "action": "explain",
            "strategy": "analogy",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"{concept} is like {analogy}. Just as [explanation of analogy], {concept} in {domain} [explanation of concept]. This helps us understand [key insight].",
            "understanding_check": {
                "required": True,
                "question": f"What analogy was used to explain {concept}, and what does it illustrate?",
                "expected_answer": f"The analogy of [analogy] was used to illustrate [key aspect of concept]."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": random.choice(["low", "medium"]),
                "misconception_addressed": random.choice(domain_info["misconceptions"]) if random.random() > 0.4 else None
            },
            "memory": {
                "candidate": random.random() > 0.6,
                "title": f"Analogy for {concept}",
                "content": f"Remember: {concept} is like [analogy].",
                "anchor_concept": concept,
                "memory_type": "analogy",
                "confidence": round(random.uniform(0.7, 0.9), 2)
            }
        }
    }

def create_concrete_example(domain: str, index: int) -> Dict[str, Any]:
    """Create a concrete example example."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    # Concrete examples
    examples = {
        "economics": [["If you earn $1000 and spend $700, your MPC is 0.7", "When gas prices rise from $3 to $4/gallon, quantity demanded might drop 10%"]],
        "mathematics": [["Solving 2x² + 5x - 3 = 0 gives x = 0.5 or x = -3", "The derivative of x³ at x=2 is 12"]],
        "science": [["In photosynthesis, 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂", "A 60kg person on Earth weighs about 10kg on the Moon"]],
        "nutrition_food_science": [["An apple contains about 95 calories and 25g of carbohydrates", "Cooking an egg denatures its proteins, making it solid"]],
        "accountancy": [["If assets = $100,000 and liabilities = $60,000, then equity = $40,000", "Revenue of $50,000 minus expenses of $30,000 equals profit of $20,000"]],
        "general_academic": [["If 60% of students passed and there are 100 students, then 40 failed", "In a debate, claiming 'we should do X because everyone does it' is a bandwagon fallacy"]]
    }

    example_list = examples.get(domain, [["For example, consider a simple case...", "In practice, this works like..."]])
    example = random.choice(example_list)

    return {
        "id": generate_example_id("concrete_example", domain, index),
        "category": "concrete_example",
        "domain": domain,
        "instruction": f"Provide a concrete example of {concept}.",
        "context": create_base_context(domain, concept),
        "target": {
            "action": "explain",
            "strategy": "concrete_example",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"Here's a concrete example: {example[0]}. This illustrates {concept} because [explanation]. Another example: {example[1]} shows [different aspect].",
            "understanding_check": {
                "required": True,
                "question": f"Based on the examples provided, what is {concept}?",
                "expected_answer": f"{concept} is [correct explanation based on examples]."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": random.choice(["low", "medium"]),
                "misconception_addressed": "Belief that abstract concepts cannot be understood without examples"
            },
            "memory": {
                "candidate": random.random() > 0.5,
                "title": f"Example of {concept}",
                "content": f"Key example: {example[0]}. Remember that [important point].",
                "anchor_concept": concept,
                "memory_type": "example",
                "confidence": round(random.uniform(0.6, 0.8), 2)
            }
        }
    }

def create_numerical_example(domain: str, index: int) -> Dict[str, Any]:
    """Create a numerical example example."""
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
    example = random.choice(example_list)

    return {
        "id": generate_example_id("numerical_example", domain, index),
        "category": "numerical_example",
        "domain": domain,
        "instruction": f"Provide a numerical example of {concept}.",
        "context": create_base_context(domain, concept),
        "target": {
            "action": "explain",
            "strategy": "concrete_example",  # Still using concrete_example strategy but could be numerical_example
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"Here's a numerical example: {example[0]}. This shows {concept} in action because [explanation]. Another case: {example[1]} demonstrates [different numerical aspect].",
            "understanding_check": {
                "required": True,
                "question": f"Based on the numerical examples provided, what does {concept} demonstrate?",
                "expected_answer": f"{concept} demonstrates [correct numerical explanation based on examples]."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": random.choice(["low", "medium"]),
                "misconception_addressed": "Belief that numerical examples aren't helpful for understanding abstract concepts"
            },
            "memory": {
                "candidate": random.random() > 0.5,
                "title": f"Numerical example of {concept}",
                "content": f"Key numerical example: {example[0]}. Remember that [important numerical point].",
                "anchor_concept": concept,
                "memory_type": "example",
                "confidence": round(random.uniform(0.6, 0.8), 2)
            }
        }
    }

def create_prerequisite_detection(domain: str, index: int) -> Dict[str, Any]:
    """Create a prerequisite detection example."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])
    missing_prereq = random.choice([p for p in domain_info["prerequisites"] if p not in ["basic algebra", "algebra"] or random.random() > 0.7])

    return {
        "id": generate_example_id("prerequisite_detection", domain, index),
        "category": "prerequisite_detection",
        "domain": domain,
        "instruction": f"Identify what prerequisite knowledge is needed to understand {concept}.",
        "context": create_base_context(domain, concept, "beginner"),
        "target": {
            "action": "explain",
            "strategy": "prerequisite",
            "difficulty": "beginner",
            "response": f"To understand {concept}, you need to know about {missing_prereq}. Without this foundation, you'll struggle because [explanation of why it's needed].",
            "understanding_check": {
                "required": True,
                "question": f"What prerequisite is essential for understanding {concept}?",
                "expected_answer": f"{missing_prereq} is essential because [reason]."
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
    """Create a misconception correction example."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])
    misconception = random.choice(domain_info["misconceptions"])

    return {
        "id": generate_example_id("misconception_detection", domain, index),
        "category": "misconception_detection",
        "domain": domain,
        "instruction": f"Address a common misconception about {concept}.",
        "context": create_base_context(domain, concept),
        "target": {
            "action": "explain",
            "strategy": "misconception_correction",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"A common misconception is that '{misconception}'. This is incorrect because [explanation of why it's wrong]. The correct understanding is that [correct explanation].",
            "understanding_check": {
                "required": True,
                "question": f"Why is the statement '{misconception}' incorrect regarding {concept}?",
                "expected_answer": f"It's incorrect because [correct explanation]."
            },
            "diagnosis": {
                "confusion_type": "misconception",
                "severity": "medium",
                "misconception_addressed": misconception
            },
            "memory": {
                "candidate": random.random() > 0.6,
                "title": f"Misconception: {concept}",
                "content": f"Remember: {misconception} is WRONG because [explanation].",
                "anchor_concept": concept,
                "memory_type": "misconception_correction",
                "confidence": round(random.uniform(0.7, 0.9), 2)
            }
        }
    }

def create_follow_up_questions(domain: str, index: int) -> Dict[str, Any]:
    """Create a follow-up questions example."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    return {
        "id": generate_example_id("follow_up_questions", domain, index),
        "category": "follow_up_questions",
        "domain": domain,
        "instruction": f"Generate follow-up questions to check understanding of {concept}.",
        "context": create_base_context(domain, concept),
        "target": {
            "action": "explain",
            "strategy": "follow_up_questions",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"To check your understanding of {concept}: [explanation]. Consider these questions: 1) [question 1]? 2) [question 2]? 3) [question 3]?",
            "understanding_check": {
                "required": True,
                "question": f"What is a good follow-up question to test understanding of {concept}?",
                "expected_answer": f"A good question would be [specific, relevant question] because [reason]."
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
    """Create a multi-turn tutoring example."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    return {
        "id": generate_example_id("multi_turn_tutoring", domain, index),
        "category": "multi_turn_tutoring",
        "domain": domain,
        "instruction": f"Continue the tutoring conversation about {concept}.",
        "context": {
            **create_base_context(domain, concept),
            "conversation_context": [
                {"role": "student", "content": f"I don't understand {concept}."},
                {"role": "tutor", "content": f"Let me explain {concept} simply: [basic explanation]."},
                {"role": "student", "content": f"I get the basic idea, but how does [specific aspect] work?"}
            ],
            "current_goal": f"Deepen understanding of {concept} through guided dialogue"
        },
        "target": {
            "action": "explain",
            "strategy": "multi_turn_tutoring",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"Great question! [Specific aspect] works by [explanation]. To see this in practice, [example/application].",
            "understanding_check": {
                "required": True,
                "question": f"Based on our conversation, how would you explain [specific aspect] of {concept}?",
                "expected_answer": f"[Specific aspect] works by [correct explanation]."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "low",
                "misconception_addressed": None
            },
            "memory": {
                "candidate": random.random() > 0.5,
                "title": f"Multi-turn insight: {concept}",
                "content": f"From our discussion: [key insight from conversation].",
                "anchor_concept": concept,
                "memory_type": "insight",
                "confidence": round(random.uniform(0.6, 0.8), 2)
            }
        }
    }

def create_repeated_confusion(domain: str, index: int) -> Dict[str, Any]:
    """Create a repeated confusion example."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    return {
        "id": generate_example_id("repeated_confusion", domain, index),
        "category": "repeated_confusion",
        "domain": domain,
        "instruction": f"Address repeated confusion about {concept}.",
        "context": {
            **create_base_context(domain, concept),
            "convenience_context": [
                {"role": "student", "content": f"I don't understand {concept}."},
                {"role": "tutor", "content": f"Let me explain {concept}: [first explanation]."},
                {"role": "student", "content": f"I still don't get it."},
                {"role": "tutor", "content": f"Let me try another way: [second explanation]."},
                {"role": "student", "content": f"I'm still confused."}
            ],
            "current_goal": f"Help student overcome persistent confusion about {concept}"
        },
        "target": {
            "action": "explain",
            "strategy": "simple_explanation",
            "difficulty": "beginner",
            "response": f"I appreciate your persistence. Let me try a different approach with {concept}: [third explanation using different strategy/examples]. The key is to focus on [core idea] rather than [common point of confusion].",
            "understanding_check": {
                "required": True,
                "question": f"What was the key breakthrough in understanding {concept}?",
                "expected_answer": f"The key was realizing [core insight]."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "medium",
                "misconception_addressed": "Belief that if you don't understand immediately, you never will"
            },
            "memory": {
                "candidate": random.random() > 0.6,
                "title": f"Overcoming confusion about {concept}",
                "content": f"Remember: When confused about {concept}, try [strategy].",
                "anchor_concept": concept,
                "memory_type": "strategy",
                "confidence": round(random.uniform(0.7, 0.9), 2)
            }
        }
    }

def create_strategy_switching(domain: str, index: int) -> Dict[str, Any]:
    """Create a strategy switching example."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    return {
        "id": generate_example_id("strategy_switching", domain, index),
        "category": "strategy_switching",
        "domain": domain,
        "instruction": f"Switch teaching strategies to better explain {concept}.",
        "context": {
            **create_base_context(domain, concept),
            "conversation_context": [
                {"role": "student", "content": f"I don't understand {concept} even after the explanation."},
                {"role": "tutor", "content": f"Let me try explaining {concept} differently."}
            ],
            "current_goal": f"Find an effective way to explain {concept} to this learner"
        },
        "target": {
            "action": "explain",
            "strategy": "analogy",  # Switching to analogy after failed straightforward explanation
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"Let me try a different approach. Think of {concept} like [analogy]. Just as [explanation of analogy], we can see that {concept} in {domain} [explanation]. This perspective helps because [reason].",
            "understanding_check": {
                "required": True,
                "question": f"Why did switching to an analogy help explain {concept}?",
                "expected_answer": f"The analogy helped by [reason related to making abstract concept concrete]."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "medium",
                "misconception_addressed": "Belief that repeating the same explanation will eventually work"
            },
            "memory": {
                "candidate": random.random() > 0.6,
                "title": f"Strategy switch for {concept}",
                "content": f"When standard explanations don't work for {concept}, try [alternative strategy].",
                "anchor_concept": concept,
                "memory_type": "strategy",
                "confidence": round(random.uniform(0.7, 0.9), 2)
            }
        }
    }

def create_diagnostic_questions(domain: str, index: int) -> Dict[str, Any]:
    """Create a diagnostic questions example."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

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
    """Create a hint-based teaching example."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    return {
        "id": generate_example_id("hint_based_teaching", domain, index),
        "category": "hint_based_teaching",
        "domain": domain,
        "instruction": f"Provide hints to help the learner figure out {concept} themselves.",
        "context": create_base_context(domain, concept),
        "target": {
            "action": "explain",
            "strategy": "hint",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"Instead of giving you the answer directly, let me give you some hints about {concept}: Hint 1: [hint 1]. Hint 2: [hint 2]. Hint 3: [hint 3]. See if you can put these together to understand {concept}.",
            "understanding_check": {
                "required": True,
                "question": f"Based on the hints provided, what is {concept}?",
                "expected_answer": f"[Correct explanation synthesizing the hints]."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "medium",
                "misconception_addressed": "Belief that learning requires direct instruction rather than discovery"
            },
            "memory": {
                "candidate": random.random() > 0.5,
                "title": f"Hint-based approach to {concept}",
                "content": f"Remember the hints for {concept}: [hint 1], [hint 2], [hint 3].",
                "anchor_concept": concept,
                "memory_type": "hint",
                "confidence": round(random.uniform(0.6, 0.8), 2)
            }
        }
    }

def create_practice_questions(domain: str, index: int) -> Dict[str, Any]:
    """Create a practice questions example."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    return {
        "id": generate_example_id("practice_questions", domain, index),
        "category": "practice_questions",
        "domain": domain,
        "instruction": f"Provide practice problems to reinforce understanding of {concept}.",
        "context": create_base_context(domain, concept),
        "target": {
            "action": "explain",
            "strategy": "practice",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"Let's practice {concept} with some exercises:\n\nProblem 1: [description]\n\nProblem 2: [description]\n\nProblem 3: [description]\n\nTry solving these, and I'll check your work.",
            "understanding_check": {
                "required": True,
                "question": f"What is the key skill demonstrated in solving these {concept} problems?",
                "expected_answer": f"The key skill is [correct skill] because [reason]."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "low",
                "misconception_addressed": None
            },
            "memory": {
                "candidate": random.random() > 0.5,
                "title": f"Practice problems for {concept}",
                "content": f"Key practice types for {concept}: [list of problem types].",
                "anchor_concept": concept,
                "memory_type": "procedure",
                "confidence": round(random.uniform(0.6, 0.8), 2)
            }
        }
    }

def create_error_correction(domain: str, index: int) -> Dict[str, Any]:
    """Create an error correction example."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    return {
        "id": generate_example_id("error_correction", domain, index),
        "category": "error_correction",
        "domain": domain,
        "instruction": f"Correct an error in the student's understanding of {concept}.",
        "context": {
            **create_base_context(domain, concept),
            "learner_state": {
                **create_base_context(domain, concept)["learner_state"],
                "known_misconceptions": [random.choice(domain_info["misconceptions"])]  # Simulate they have this misconception
            }
        },
        "target": {
            "action": "explain",
            "strategy": "misconception_correction",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"I notice there's a misunderstanding in your work. You stated that '[incorrect statement related to misconception]', but that's not quite right. Let me correct this: [correct explanation]. The error likely came from [likely source of confusion].",
            "understanding_check": {
                "required": True,
                "question": f"What was the error in thinking about {concept}, and what is the correct understanding?",
                "expected_answer": f"The error was [incorrect idea], and the correct understanding is [correct idea]."
            },
            "diagnosis": {
                "confusion_type": "misconception",
                "severity": "medium",
                "misconception_addressed": random.choice(domain_info["misconceptions"])
            },
            "memory": {
                "candidate": random.random() > 0.6,
                "title": f"Corrected understanding: {concept}",
                "content": f"Remember: [correct explanation], NOT [incorrect idea].",
                "anchor_concept": concept,
                "memory_type": "misconception_correction",
                "confidence": round(random.uniform(0.7, 0.9), 2)
            }
        }
    }

def create_partial_understanding(domain: str, index: int) -> Dict[str, Any]:
    """Create a partial understanding example."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    return {
        "id": generate_example_id("partial_understanding", domain, index),
        "category": "partial_understanding",
        "domain": domain,
        "instruction": f"Address partial understanding of {concept}.",
        "context": create_base_context(domain, concept),
        "target": {
            "action": "explain",
            "strategy": "partial_understanding",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"You've got part of {concept} right - you correctly identified that [correct aspect]. However, you're missing [missing aspect]. Let me explain the full picture: [complete explanation].",
            "understanding_check": {
                "required": True,
                "question": f"What aspect of {concept} was missing from the initial understanding?",
                "expected_answer": f"The missing aspect was [missing aspect] because [reason]."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "low",
                "misconception_addressed": None
            },
            "memory": {
                "candidate": random.random() > 0.5,
                "title": f"Complete understanding of {concept}",
                "content": f"To fully understand {concept}, remember both [aspect 1] and [aspect 2].",
                "anchor_concept": concept,
                "memory_type": "concept",
                "confidence": round(random.uniform(0.6, 0.8), 2)
            }
        }
    }

def create_understanding_confirmation(domain: str, index: int) -> Dict[str, Any]:
    """Create an understanding confirmation example."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    return {
        "id": generate_example_id("understanding_confirmation", domain, index),
        "category": "understanding_confirmation",
        "domain": domain,
        "instruction": f"Confirm the student's understanding of {concept}.",
        "context": {
            **create_base_context(domain, concept),
            "learner_state": {
                **create_base_context(domain, concept)["learner_state"],
                # Simulate they've been learning and now show improved mastery
                "mastery": min(0.9, create_base_context(domain, concept)["learner_state"]["mastery"] + 0.3)
            }
        },
        "target": {
            "action": "explain",
            "strategy": "understanding_confirmation",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"Excellent work! You've demonstrated solid understanding of {concept} by [specific correct action/application]. This shows you've grasped the key concepts including [key points].",
            "understanding_check": {
                "required": True,
                "question": f"How would you explain {concept} to someone else who is just learning it?",
                "expected_answer": f"I would explain it as [clear, correct explanation] because [reason]."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "very_low",  # Almost no confusion left
                "misconception_addressed": None
            },
            "memory": {
                "candidate": random.random() > 0.7,
                "title": f"Mastered concept: {concept}",
                "content": f"You now understand {concept} well enough to [application].",
                "anchor_concept": concept,
                "memory_type": "mastery",
                "confidence": round(random.uniform(0.8, 0.95), 2)
            }
        }
    }

def create_memory_generation(domain: str, index: int) -> Dict[str, Any]:
    """Create a memory generation example."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    return {
        "id": generate_example_id("memory_generation", domain, index),
        "category": "memory_generation",
        "domain": domain,
        "instruction": f"Generate a memory candidate for {concept}.",
        "context": create_base_context(domain, concept),
        "target": {
            "action": "explain",
            "strategy": "simple_explanation",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"{concept} in {domain} is defined as [clear, concise definition]. It's important because [reason]. A key example is [example].",
            "understanding_check": {
                "required": True,
                "question": f"What should we remember about {concept} in {domain}?",
                "expected_answer": f"We should remember that {concept} [key point to remember]."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "low",
                "misconception_addressed": None
            },
            "memory": {
                "candidate": True,  # This is specifically for memory generation
                "title": f"Essential: {concept}",
                "content": f"{concept} in {domain}: [concise, memorable definition]. Key point: [important implication].",
                "anchor_concept": concept,
                "memory_type": "definition",
                "confidence": round(random.uniform(0.8, 0.95), 2)
            }
        }
    }

def create_document_grounded_teaching(domain: str, index: int) -> Dict[str, Any]:
    """Create a document-grounded teaching example."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    return {
        "id": generate_example_id("document_grounded", domain, index),
        "category": "document_grounded",
        "domain": domain,
        "instruction": f"Teach {concept} using specific evidence from the document.",
        "context": create_base_context(domain, concept),
        "target": {
            "action": "explain",
            "strategy": "simple_explanation",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"Looking at our document, we can see that {concept} is clearly explained in the selected text: '[quote from selected text]'. The surrounding context tells us that [explanation from surrounding context]. The retrieved evidence supports this by showing [evidence explanation].",
            "understanding_check": {
                "required": True,
                "question": f"What evidence from the document supports the explanation of {concept}?",
                "expected_answer": f"The selected text states '[quote]', and the surrounding context adds [explanation]."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "low",
                "misconception_addressed": None
            },
            "memory": {
                "candidate": random.random() > 0.5,
                "title": f"Document-based insight: {concept}",
                "content": f"From our reading: {concept} means [explanation based on document].",
                "anchor_concept": concept,
                "memory_type": "document_insight",
                "confidence": round(random.uniform(0.6, 0.8), 2)
            }
        }
    }

def create_context_retention(domain: str, index: int) -> Dict[str, Any]:
    """Create a context retention example."""
    domain_info = DOMAIN_CONTENT[domain]
    concept = random.choice(domain_info["concepts"])

    return {
        "id": generate_example_id("context_retention", domain, index),
        "category": "context_retention",
        "domain": domain,
        "instruction": f"Demonstrate retention of earlier context while discussing {concept}.",
        "context": {
            **create_base_context(domain, concept),
            "conversation_context": [
                {"role": "student", "content": f"Earlier we talked about [different concept]. Now I'm confused about {concept}."},
                {"role": "tutor", "content": f"Let's connect [different concept] to {concept}."}
            ],
            "current_goal": f"Show how {concept} relates to previously discussed material"
        },
        "target": {
            "action": "explain",
            "strategy": "simple_explanation",
            "difficulty": random.choice(["beginner", "intermediate"]),
            "response": f"Great question about connecting [different concept] to {concept}! Earlier we established that [fact about different concept]. This relates to {concept} because [connection explanation]. Now, to understand {concept} specifically: [explanation of concept].",
            "understanding_check": {
                "required": True,
                "question": f"How does [different concept] relate to {concept}?",
                "expected_answer": f"[Different concept] relates to {concept} because [correct explanation of connection]."
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "low",
                "misconception_addressed": None
            },
            "memory": {
                "candidate": random.random() > 0.5,
                "title": f"Connecting concepts: {concept} and [different concept]",
                "content": f"Remember: {concept} and [different concept] are related because [explanation].",
                "anchor_concept": concept,
                "memory_type": "connection",
                "confidence": round(random.uniform(0.6, 0.8), 2)
            }
        }
    }

def generate_all_categories(domain: str, count_per_category: int) -> List[Dict[str, Any]]:
    """Generate examples for all categories in a domain."""
    examples = []

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

    # Generate examples for each category
    for category, generator_func in generators.items():
        for i in range(count_per_category):
            example = generator_func(domain, i)
            examples.append(example)

    return examples

def generate_enhanced_seed_data(output_dir: str = "./data/generated",
                               examples_per_domain: int = 20) -> List[Dict[str, Any]]:
    """
    Generate enhanced seed data for OLORIC.

    Args:
        output_dir: Directory to save generated data
        examples_per_domain: Number of examples per category per domain

    Returns:
        List of generated training examples
    """
    print("Generating enhanced OLORIC seed data...")

    all_examples = []

    # Calculate how many examples per category to reach target
    # We have 19 categories and 6 domains
    # Target: 300-500 examples total
    # Examples per (category, domain) pair = target / (19 * 6) ≈ target / 114
    # For 400 target: 400/114 ≈ 3.5, so let's use 4 examples per (category, domain) = 456 total

    for domain in VALID_DOMAINS:
        print(f"Generating examples for {domain}...")
        domain_examples = generate_all_categories(domain, examples_per_domain)
        all_examples.extend(domain_examples)
        print(f"  Generated {len(domain_examples)} examples")

    # Shuffle to mix domains and categories
    random.shuffle(all_examples)

    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "enhanced_seed_data.jsonl")

    print(f"Saving {len(all_examples)} examples to {output_file}")
    with open(output_file, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')

    print("Enhanced seed data generation complete!")
    return all_examples

def main():
    """Main generation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate enhanced OLORIC seed data")
    parser.add_argument("--output-dir", type=str, default="./data/generated",
                        help="Output directory for generated data")
    parser.add_argument("--examples-per-category", type=int, default=4,
                        help="Number of examples per category per domain")
    parser.add_argument("--total-target", type=int, default=400,
                        help="Target total number of examples")

    args = parser.parse_args()

    examples = generate_enhanced_seed_data(
        output_dir=args.output_dir,
        examples_per_domain=args.examples_per_category
    )

    print(f"\nGenerated {len(examples)} examples saved to: {os.path.join(args.output_dir, 'enhanced_seed_data.jsonl')}")

    # Quick statistics
    categories_count = {}
    domains_count = {}

    for example in examples:
        cat = example["category"]
        dom = example["domain"]
        categories_count[cat] = categories_count.get(cat, 0) + 1
        domains_count[dom] = domains_count.get(dom, 0) + 1

    print("\nGeneration Statistics:")
    print(f"Total examples: {len(examples)}")
    print(f"Categories: {len(categories_count)} different categories")
    print(f"Domains: {len(domains_count)} different domains")

    print("\nExamples per category:")
    for category, count in sorted(categories_count.items()):
        print(f"  {category}: {count}")

    print("\nExamples per domain:")
    for domain, count in sorted(domains_count.items()):
        print(f"  {domain}: {count}")

    return 0

if __name__ == "__main__":
    exit(main())