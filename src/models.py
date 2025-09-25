"""
Data models for HANRAG system.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


class DocumentType(str, Enum):
    """Types of documents in the knowledge base."""

    TEXT = "text"
    WIKIPEDIA = "wikipedia"
    SCIENTIFIC_PAPER = "scientific_paper"
    NEWS_ARTICLE = "news_article"
    DEFINITION = "definition"
    SCIENCE = "science"


class QueryType(str, Enum):
    """Types of queries as defined in the HANRAG paper."""

    STRAIGHTFORWARD = "straightforward"
    SINGLE_STEP = "single_step"
    COMPOUND = "compound"
    COMPLEX = "complex"


class RetrievalResult(BaseModel):
    """Result from document retrieval."""

    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    document_type: DocumentType = DocumentType.TEXT


class ReasoningStep(BaseModel):
    """A single step in multi-hop reasoning."""

    step_number: int = 1
    question: str = ""
    retrieved_documents: List[RetrievalResult] = Field(default_factory=list)
    reasoning: str = ""
    intermediate_answer: Optional[str] = None
    confidence: float = 0.0


class MultiHopQuery(BaseModel):
    """A multi-hop question with reasoning steps."""

    original_question: str = ""
    reasoning_steps: List[ReasoningStep] = Field(default_factory=list)
    final_answer: Optional[str] = None
    confidence: float = 0.0
    noise_detected: bool = False
    noise_level: float = 0.0


class HANRAGResponse(BaseModel):
    """Final response from HANRAG system."""

    query: MultiHopQuery
    answer: str
    confidence: float
    reasoning_chain: List[ReasoningStep]
    retrieved_documents: List[RetrievalResult]
    processing_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NoiseDetectionResult(BaseModel):
    """Result from noise detection analysis."""

    is_noisy: bool
    noise_score: float
    noise_type: Optional[str] = None
    confidence: float
    explanation: str


class HeuristicRule(BaseModel):
    """A heuristic rule for improving retrieval accuracy."""

    rule_id: str
    name: str
    description: str
    condition: str
    action: str
    weight: float = 1.0
    active: bool = True
