"""
Input schema for the OLORIC model.
Defines the structured input format that the model expects.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DocumentContext(BaseModel):
    """Context about the document being studied."""
    document_id: str = Field(..., description="Unique identifier for the document")
    title: str = Field(..., description="Title of the document")
    page: int = Field(..., description="Page number where confusion occurred")
    section: str = Field(..., description="Section title or identifier")
    selected_text: str = Field(..., description="Specific text the student is confused about")
    surrounding_context: str = Field(..., description="Broader context around the selected text")
    retrieved_evidence: List[str] = Field(
        default_factory=list,
        description="Additional retrieved evidence or snippets from RAG"
    )


class LearnerState(BaseModel):
    """Current state of the learner."""
    level: str = Field(..., description="Proficiency level (beginner, intermediate, advanced)")
    concept: str = Field(..., description="Current concept being studied")
    mastery: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Mastery score for the concept (0.0 to 1.0)"
    )
    known_prerequisites: List[str] = Field(
        default_factory=list,
        description="Prerequisites the learner has mastered"
    )
    weak_prerequisites: List[str] = Field(
        default_factory=list,
        description="Prerequisites the learner struggles with"
    )
    known_misconceptions: List[str] = Field(
        default_factory=list,
        description="Misconceptions the learner is known to have"
    )


class ConversationTurn(BaseModel):
    """Single turn in the conversation."""
    role: str = Field(..., description="Speaker role (student or tutor)")
    content: str = Field(..., description="Content of the utterance")


class OloricModelInput(BaseModel):
    """Complete input schema for the OLORIC model."""
    task: str = Field(
        default="resolve_confusion",
        description="Current task to perform"
    )
    document_context: DocumentContext = Field(..., description="Context about the document")
    learner_state: LearnerState = Field(..., description="Current state of the learner")
    conversation_context: List[ConversationTurn] = Field(
        default_factory=list,
        description="History of the conversation so far"
    )
    current_goal: str = Field(
        ..., 
        description="What the learner wants to achieve"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
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
                    }
                ],
                "current_goal": "Understand MPC well enough to apply it"
            }
        }
