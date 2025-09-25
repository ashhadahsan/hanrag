"""
HANRAG: Heuristic Accurate Noise-resistant Retrieval-Augmented Generation
for Multi-hop Question Answering

This package implements the HANRAG system as described in the research paper.
"""

from .hanrag import HANRAGSystem
from .revelator import Revelator
from .models import (
    MultiHopQuery,
    HANRAGResponse,
    RetrievalResult,
    ReasoningStep,
    NoiseDetectionResult,
    DocumentType,
    HeuristicRule,
    QueryType,
)
from .retrieval import NoiseResistantRetriever, HeuristicRetriever
from .generation import NoiseResistantGenerator, AnswerGenerator
from .config import config

__version__ = "2.0.0"
__author__ = "HANRAG Implementation Team"

__all__ = [
    "HANRAGSystem",
    "Revelator",
    "MultiHopQuery",
    "HANRAGResponse",
    "RetrievalResult",
    "ReasoningStep",
    "NoiseDetectionResult",
    "DocumentType",
    "HeuristicRule",
    "QueryType",
    "NoiseResistantRetriever",
    "HeuristicRetriever",
    "NoiseResistantGenerator",
    "AnswerGenerator",
    "config",
]
