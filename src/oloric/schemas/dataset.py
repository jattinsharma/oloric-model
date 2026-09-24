"""
Dataset schemas for OLORIC.
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class TrainingExample(BaseModel):
    """Schema for a single training example in the OLORIC dataset."""

    id: str = Field(..., description="Unique identifier for the example")
    category: str = Field(..., description="Category of teaching strategy")
    domain: str = Field(..., description="Subject domain")
    instruction: str = Field(..., description="Instruction for the task")
    context: Dict[str, Any] = Field(..., description="Input context")
    target: Dict[str, Any] = Field(..., description="Expected output")
    conversation: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Conversation history"
    )

    class Config:
        """Pydantic configuration."""
        extra = "forbid"


# Valid categories for the dataset
VALID_CATEGORIES = [
    "simple_explanation", "simplification", "analogy", "concrete_example",
    "numerical_example", "prerequisite_detection", "misconception_detection",
    "follow_up_questions", "multi_turn_tutoring", "repeated_confusion",
    "strategy_switching", "diagnostic_questions", "hint_based_teaching",
    "practice_questions", "error_correction", "partial_understanding",
    "understanding_confirmation", "memory_generation", "document_grounded",
    "context_retention"
]

# Valid domains for the dataset
VALID_DOMAINS = [
    "economics", "accountancy", "mathematics", "science",
    "nutrition_food_science", "general_academic"
]