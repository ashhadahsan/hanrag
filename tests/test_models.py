"""
Tests for HANRAG data models.
"""

import pytest
from src.models import (
    DocumentType,
    RetrievalResult,
    ReasoningStep,
    MultiHopQuery,
    HANRAGResponse,
    NoiseDetectionResult,
    HeuristicRule,
)


class TestDocumentType:
    """Test DocumentType enum."""

    def test_document_type_values(self):
        """Test DocumentType enum values."""
        assert DocumentType.TEXT == "text"
        assert DocumentType.WIKIPEDIA == "wikipedia"
        assert DocumentType.SCIENTIFIC_PAPER == "scientific_paper"
        assert DocumentType.NEWS_ARTICLE == "news_article"


class TestRetrievalResult:
    """Test RetrievalResult model."""

    def test_retrieval_result_creation(self):
        """Test creating a RetrievalResult."""
        result = RetrievalResult(
            document_id="doc_1",
            content="This is a test document.",
            score=0.85,
            metadata={"source": "test"},
            document_type=DocumentType.TEXT,
        )

        assert result.document_id == "doc_1"
        assert result.content == "This is a test document."
        assert result.score == 0.85
        assert result.metadata == {"source": "test"}
        assert result.document_type == DocumentType.TEXT

    def test_retrieval_result_defaults(self):
        """Test RetrievalResult with default values."""
        result = RetrievalResult(
            document_id="doc_2", content="Another test document.", score=0.75
        )

        assert result.metadata == {}
        assert result.document_type == DocumentType.TEXT


class TestReasoningStep:
    """Test ReasoningStep model."""

    def test_reasoning_step_creation(self):
        """Test creating a ReasoningStep."""
        documents = [
            RetrievalResult(document_id="doc_1", content="Test content", score=0.8)
        ]

        step = ReasoningStep(
            step_number=1,
            question="What is the main topic?",
            retrieved_documents=documents,
            reasoning="Based on the document content...",
            intermediate_answer="The main topic is testing.",
            confidence=0.9,
        )

        assert step.step_number == 1
        assert step.question == "What is the main topic?"
        assert len(step.retrieved_documents) == 1
        assert step.reasoning == "Based on the document content..."
        assert step.intermediate_answer == "The main topic is testing."
        assert step.confidence == 0.9

    def test_reasoning_step_defaults(self):
        """Test ReasoningStep with default values."""
        step = ReasoningStep(
            step_number=2, question="Another question?", retrieved_documents=[]
        )

        assert step.intermediate_answer is None
        assert step.confidence == 0.0


class TestMultiHopQuery:
    """Test MultiHopQuery model."""

    def test_multi_hop_query_creation(self):
        """Test creating a MultiHopQuery."""
        reasoning_steps = [
            ReasoningStep(
                step_number=1,
                question="Step 1?",
                retrieved_documents=[],
                reasoning="Reasoning 1",
                confidence=0.8,
            )
        ]

        query = MultiHopQuery(
            original_question="What is the answer?",
            reasoning_steps=reasoning_steps,
            final_answer="The answer is 42.",
            confidence=0.9,
            noise_detected=True,
            noise_level=0.3,
        )

        assert query.original_question == "What is the answer?"
        assert len(query.reasoning_steps) == 1
        assert query.final_answer == "The answer is 42."
        assert query.confidence == 0.9
        assert query.noise_detected is True
        assert query.noise_level == 0.3

    def test_multi_hop_query_defaults(self):
        """Test MultiHopQuery with default values."""
        query = MultiHopQuery(original_question="Simple question?")

        assert query.reasoning_steps == []
        assert query.final_answer is None
        assert query.confidence == 0.0
        assert query.noise_detected is False
        assert query.noise_level == 0.0


class TestHANRAGResponse:
    """Test HANRAGResponse model."""

    def test_hanrag_response_creation(self):
        """Test creating a HANRAGResponse."""
        query = MultiHopQuery(original_question="Test question?")
        reasoning_steps = [
            ReasoningStep(
                step_number=1,
                question="Step 1?",
                retrieved_documents=[],
                reasoning="Test reasoning",
                confidence=0.8,
            )
        ]
        retrieved_docs = [
            RetrievalResult(document_id="doc_1", content="Test content", score=0.9)
        ]

        response = HANRAGResponse(
            query=query,
            answer="Test answer",
            confidence=0.85,
            reasoning_chain=reasoning_steps,
            retrieved_documents=retrieved_docs,
            processing_time=1.5,
            metadata={"test": "value"},
        )

        assert response.query == query
        assert response.answer == "Test answer"
        assert response.confidence == 0.85
        assert len(response.reasoning_chain) == 1
        assert len(response.retrieved_documents) == 1
        assert response.processing_time == 1.5
        assert response.metadata == {"test": "value"}

    def test_hanrag_response_defaults(self):
        """Test HANRAGResponse with default values."""
        query = MultiHopQuery(original_question="Test question?")

        response = HANRAGResponse(
            query=query,
            answer="Test answer",
            confidence=0.8,
            reasoning_chain=[],
            retrieved_documents=[],
            processing_time=1.0,
        )

        assert response.metadata == {}


class TestNoiseDetectionResult:
    """Test NoiseDetectionResult model."""

    def test_noise_detection_result_creation(self):
        """Test creating a NoiseDetectionResult."""
        result = NoiseDetectionResult(
            is_noisy=True,
            noise_score=0.7,
            noise_type="query_ambiguity",
            confidence=0.8,
            explanation="Query contains ambiguous terms",
        )

        assert result.is_noisy is True
        assert result.noise_score == 0.7
        assert result.noise_type == "query_ambiguity"
        assert result.confidence == 0.8
        assert result.explanation == "Query contains ambiguous terms"

    def test_noise_detection_result_defaults(self):
        """Test NoiseDetectionResult with default values."""
        result = NoiseDetectionResult(
            is_noisy=False,
            noise_score=0.2,
            confidence=0.9,
            explanation="Query is clear",
        )

        assert result.noise_type is None


class TestHeuristicRule:
    """Test HeuristicRule model."""

    def test_heuristic_rule_creation(self):
        """Test creating a HeuristicRule."""
        rule = HeuristicRule(
            rule_id="test_rule",
            name="Test Rule",
            description="A test heuristic rule",
            condition="score > 0.5",
            action="boost_score *= 1.2",
            weight=1.2,
            active=True,
        )

        assert rule.rule_id == "test_rule"
        assert rule.name == "Test Rule"
        assert rule.description == "A test heuristic rule"
        assert rule.condition == "score > 0.5"
        assert rule.action == "boost_score *= 1.2"
        assert rule.weight == 1.2
        assert rule.active is True

    def test_heuristic_rule_defaults(self):
        """Test HeuristicRule with default values."""
        rule = HeuristicRule(
            rule_id="default_rule",
            name="Default Rule",
            description="A default rule",
            condition="always",
            action="no_action",
        )

        assert rule.weight == 1.0
        assert rule.active is True
