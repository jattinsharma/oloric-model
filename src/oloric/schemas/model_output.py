"""
Model output schemas for OLORIC.
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class UnderstandingCheck(BaseModel):
    """Understanding check component."""
    required: bool = Field(False, description="Whether understanding check is required")
    question: Optional[str] = Field(None, description="Question to check understanding")
    expected_answer: Optional[str] = Field(None, description="Expected answer to the question")


class Diagnosis(BaseModel):
    """Diagnosis of learner's confusion."""
    confusion_type: str = Field(..., description="Type of confusion (e.g., conceptual, procedural)")
    severity: str = Field(..., description="Severity of confusion (e.g., low, medium, high)")
    misconception_addressed: Optional[str] = Field(
        None,
        description="Specific misconception being addressed"
    )


class Memory(BaseModel):
    """Memory component for potential storage."""
    candidate: bool = Field(False, description="Whether this should be stored as a memory")
    title: Optional[str] = Field(None, description="Title of the memory")
    content: Optional[str] = Field(None, description="Content of the memory")
    anchor_concept: Optional[str] = Field(
        None,
        description="Concept this memory is anchored to"
    )
    memory_type: Optional[str] = Field(
        None,
        description="Type of memory (e.g., definition, procedure, example)"
    )
    confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in this memory candidate (0.0 to 1.0)"
    )


class OloricModelOutput(BaseModel):
    """Output model for OLORIC."""
    task: Optional[str] = Field(None, description="Task performed (optional)")
    action: str = Field(..., description="Action taken (e.g., explain, simplify, analogy)")
    strategy: str = Field(..., description="Strategy used (e.g., simple_explanation)")
    difficulty: str = Field(..., description="Difficulty level (beginner, intermediate, advanced)")
    response: str = Field(..., description="Main response text")
    understanding_check: UnderstandingCheck = Field(
        default_factory=UnderstandingCheck,
        description="Understanding check component"
    )
    diagnosis: Diagnosis = Field(..., description="Diagnosis of confusion")
    memory: Memory = Field(
        default_factory=Memory,
        description="Memory component for potential storage"
    )

    class Config:
        """Pydantic configuration."""
        extra = "forbid"


# Valid actions for the model output
VALID_ACTIONS = {
    "diagnose", "explain", "simplify", "analogy", "example",
    "prerequisite", "misconception_correction", "hint", "practice",
    "recap", "confirm_understanding", "memory_candidate"
}

# Valid difficulty levels
VALID_DIFFICULTIES = {"beginner", "intermediate", "advanced"}