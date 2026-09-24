"""
Schema for OLORIC training dataset.
Defines the format for training examples.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from .model_input import OloricModelInput
from .model_output import OloricModelOutput


class TrainingExample(BaseModel):
    """Single training example for OLORIC."""
    id: str = Field(..., description="Unique identifier for the example")
    category: str = Field(..., description="Category of tutoring behavior")
    domain: str = Field(..., description="Academic domain (economics, math, etc.)")
    instruction: str = Field(..., description="Instruction or task description")
    context: OloricModelInput = Field(..., description="Full input context")
    target: OloricModelOutput = Field(..., description="Expected output")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "id": "example_001",
                "category": "strategy_switch",
                "domain": "economics",
                "instruction": "Resolve student confusion about MPC",
                "context": {
                    "task": "resolve_confusion",
                    "document_context": {
                        "document_id": "econ101_chapter3",
                        "title": "Principles of Economics",
                        "page": 37,
                        "section": "Multiplier",
                        "selected_text": "The marginal propensity to consume (MPC)...",
                        "surrounding_context": "In Keynesian economics, the multiplier effect...",
                        "retrieved_evidence": [
                            "MPC = ΔC/ΔY where C is consumption and Y is income",
                            "The multiplier is 1/(1-MPC)"
                        ]
                    },
                    "learner_state": {
                        "level": "beginner",
                        "concept": "MPC",
                        "mastery": 0.42,
                        "known_prerequisites": ["basic algebra"],
                        "weak_prerequisites": ["understanding of slopes"],
                        "known_misconceptions": []
                    },
                    "conversation_context": [
                        {
                            "role": "student",
                            "content": "I don't understand MPC."
                        },
                        {
                            "role": "tutor",
                            "content": "MPC, or marginal propensity to consume, is how much extra spending happens when someone gets extra income. If you get $1 extra and spend 80 cents of it, your MPC is 0.8."
                        }
                    ],
                    "current_goal": "Understand MPC well enough to apply it"
                },
                "target": {
                    "action": "explain",
                    "strategy": "numerical_example",
                    "difficulty": "beginner",
                    "response": "Let's look at a numerical example. If you receive a $100 bonus and your MPC is 0.8, you will spend $80 of that bonus and save $20. This $80 spending becomes income for others, who then spend 80% of it ($64), and so on.",
                    "understanding_check": {
                        "required": True,
                        "question": "If your MPC is 0.6 and you get $200 extra income, how much will you spend?"
                    },
                    "diagnosis": {
                        "confusion_type": "conceptual",
                        "misconception": None,
                        "missing_prerequisite": None
                    },
                    "memory": {
                        "candidate": False,
                        "title": None,
                        "content": None,
                        "anchor_concept": None
                    }
                }
            }
        }
