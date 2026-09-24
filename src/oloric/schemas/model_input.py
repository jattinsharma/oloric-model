"""
Model input schemas for OLORIC.
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class DocumentContext(BaseModel):
    """Context about the document being studied."""
    document_id: str = Field(..., description="Unique identifier for the document")
    title: str = Field(..., description="Title of the document")
    page: int = Field(..., description="Page number")
    section: str = Field(..., description="Section or chapter")
    selected_text: str = Field(..., description="Specific text selected by the learner")
    surrounding_context: str = Field(..., description="Surrounding text context")
    retrieved_evidence: List[str] = Field(
        default_factory=list,
        description="Evidence retrieved to support explanation"
    )


class LearnerState(BaseModel):
    """Current state of the learner."""
    level: str = Field(..., description="Learner level (beginner, intermediate, advanced)")
    concept: str = Field(..., description="Current concept being studied")
    mastery: float = Field(..., ge=0.0, le=1.0, description="Mastery level (0.0 to 1.0)")
    known_prerequisites: List[str] = Field(
        default_factory=list,
        description="Prerequisites the learner already knows"
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
    """A single turn in a conversation."""
    role: str = Field(..., description="Role of speaker (student or tutor)")
    content: str = Field(..., description="Content of the utterance")


class OloricModelInput(BaseModel):
    """Input model for OLORIC."""
    task: str = Field(..., description="Task to perform (e.g., resolve_confusion)")
    document_context: DocumentContext = Field(..., description="Context about the document")
    learner_state: LearnerState = Field(..., description="Current state of the learner")
    conversation_context: List[ConversationTurn] = Field(
        default_factory=list,
        description="History of conversation"
    )
    current_goal: str = Field(..., description="Current learning goal")

    class Config:
        """Pydantic configuration."""
        # extra = "ignore" allows the audited dataset convenience_context field
        # to pass validation. Only declared fields are used at inference time.
        extra = "ignore"