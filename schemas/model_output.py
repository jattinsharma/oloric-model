"""
Output schema for the OLORIC model.
Defines the structured output format that the model produces.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class UnderstandingCheck(BaseModel):
    """Check for learner understanding."""
    required: bool = Field(..., description="Whether understanding check is needed")
    question: Optional[str] = Field(
        default=None,
        description="Question to check understanding (if required)"
    )


class Diagnosis(BaseModel):
    """Diagnosis of the learner's confusion."""
    confusion_type: str = Field(
        ..., 
        description="Type of confusion (conceptual, procedural, vocabulary, etc.)"
    )
    misconception: Optional[str] = Field(
        default=None,
        description="Identified misconception, if any"
    )
    missing_prerequisite: Optional[str] = Field(
        default=None,
        description="Missing prerequisite, if any"
    )


class MemoryCandidate(BaseModel):
    """Potential memory to be saved by the application."""
    candidate: bool = Field(..., description="Whether this is a memory candidate")
    title: Optional[str] = Field(
        default=None,
        description="Title for the memory (if candidate)"
    )
    content: Optional[str] = Field(
        default=None,
        description="Content for the memory (if candidate)"
    )
    anchor_concept: Optional[str] = Field(
        default=None,
        description="Concept to anchor the memory to (if candidate)"
    )


class OloricModelOutput(BaseModel):
    """Complete output schema for the OLORIC model."""
    action: str = Field(..., description="Tutoring action to perform")
    strategy: str = Field(..., description="Teaching strategy to use")
    difficulty: str = Field(..., description="Difficulty level of the response")
    response: str = Field(..., description="The actual tutoring response")
    understanding_check: UnderstandingCheck = Field(..., description="Understanding check details")
    diagnosis: Diagnosis = Field(..., description="Diagnosis of confusion")
    memory: MemoryCandidate = Field(..., description="Memory candidate information")

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "action": "explain",
                "strategy": "simple_example",
                "difficulty": "beginner",
                "response": "MPC, or marginal propensity to consume, is how much extra spending happens when someone gets extra income. If you get $1 extra and spend 80 cents of it, your MPC is 0.8.",
                "understanding_check": {
                    "required": True,
                    "question": "If someone gets $100 extra and spends $75 of it, what is their MPC?"
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
        
        # Allowed action values
        _allowed_actions = {
            "diagnose", "explain", "simplify", "analogy", "example", 
            "prerequisite", "misconception_correction", "hint", "practice", 
            "recap", "confirm_understanding", "memory_candidate"
        }
