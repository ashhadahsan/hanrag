"""
Tests for HANRAG generation components.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

from src.models import (
    RetrievalResult,
    ReasoningStep,
    MultiHopQuery,
    HANRAGResponse,
    NoiseDetectionResult,
)
from src.generation import AnswerGenerator, NoiseResistantGenerator


class TestAnswerGenerator:
    """Test AnswerGenerator class."""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing."""
        mock_llm = Mock(spec=ChatOpenAI)
        mock_llm.invoke.return_value = AIMessage(content="Test response")
        return mock_llm

    @pytest.fixture
    def sample_retrieval_results(self):
        """Sample retrieval results for testing."""
        return [
            RetrievalResult(
                document_id="doc1",
                content="Machine learning is a subset of artificial intelligence.",
                score=0.9,
                metadata={"source": "test"},
            ),
            RetrievalResult(
                document_id="doc2",
                content="Deep learning uses neural networks with multiple layers.",
                score=0.8,
                metadata={"source": "test"},
            ),
        ]

    @patch("src.generation.ChatOpenAI")
    def test_answer_generator_initialization(self, mock_chat_openai, mock_llm):
        """Test AnswerGenerator initialization."""
        mock_chat_openai.return_value = mock_llm

        generator = AnswerGenerator()

        assert generator.model_name is not None
        assert generator.llm is not None
        assert generator.reasoning_prompt is not None
        assert generator.answer_prompt is not None
        assert generator.confidence_prompt is not None

    @patch("src.generation.ChatOpenAI")
    def test_generate_reasoning_steps(
        self, mock_chat_openai, mock_llm, sample_retrieval_results
    ):
        """Test reasoning step generation."""
        mock_chat_openai.return_value = mock_llm

        # Mock LLM response for reasoning
        reasoning_response = AIMessage(
            content="""
        Step 1: What is machine learning?
        Reasoning: Based on the documents, I need to understand what machine learning is.
        Answer: Machine learning is a subset of artificial intelligence.
        Confidence: 0.9
        
        Step 2: How does it relate to deep learning?
        Reasoning: I need to connect machine learning to deep learning concepts.
        Answer: Deep learning is a subset of machine learning that uses neural networks.
        Confidence: 0.8
        """
        )
        mock_llm.invoke.return_value = reasoning_response

        generator = AnswerGenerator()
        steps = generator._generate_reasoning_steps(
            "What is machine learning?", sample_retrieval_results
        )

        assert len(steps) == 2
        assert all(isinstance(step, ReasoningStep) for step in steps)
        assert steps[0].step_number == 1
        assert steps[1].step_number == 2
        assert steps[0].confidence == 0.9
        assert steps[1].confidence == 0.8

    @patch("src.generation.ChatOpenAI")
    def test_parse_reasoning_steps(
        self, mock_chat_openai, mock_llm, sample_retrieval_results
    ):
        """Test parsing of reasoning steps from LLM response."""
        mock_chat_openai.return_value = mock_llm

        generator = AnswerGenerator()

        reasoning_text = """
        Step 1: What is machine learning?
        Reasoning: Based on the documents provided.
        Answer: Machine learning is a subset of AI.
        Confidence: 0.9
        
        Step 2: How does it work?
        Reasoning: It uses algorithms to learn patterns.
        Answer: It learns from data automatically.
        Confidence: 0.8
        """

        steps = generator._parse_reasoning_steps(
            reasoning_text, sample_retrieval_results
        )

        assert len(steps) == 2
        assert steps[0].question == "What is machine learning?"
        assert steps[0].reasoning == "Based on the documents provided."
        assert steps[0].intermediate_answer == "Machine learning is a subset of AI."
        assert steps[0].confidence == 0.9

        assert steps[1].question == "How does it work?"
        assert steps[1].reasoning == "It uses algorithms to learn patterns."
        assert steps[1].intermediate_answer == "It learns from data automatically."
        assert steps[1].confidence == 0.8

    @patch("src.generation.ChatOpenAI")
    def test_generate_final_answer(
        self, mock_chat_openai, mock_llm, sample_retrieval_results
    ):
        """Test final answer generation."""
        mock_chat_openai.return_value = mock_llm

        # Mock LLM response for final answer
        final_answer_response = AIMessage(
            content="""
        Final Answer: Machine learning is a subset of artificial intelligence that enables computers to learn from data.
        Confidence: 0.9
        Evidence: The documents clearly state that machine learning is a subset of AI.
        """
        )
        mock_llm.invoke.return_value = final_answer_response

        generator = AnswerGenerator()

        reasoning_steps = [
            ReasoningStep(
                step_number=1,
                question="What is machine learning?",
                retrieved_documents=sample_retrieval_results,
                reasoning="Based on the documents",
                intermediate_answer="Machine learning is a subset of AI",
                confidence=0.9,
            )
        ]

        final_answer, confidence = generator._generate_final_answer(
            "What is machine learning?", reasoning_steps, sample_retrieval_results
        )

        assert "Machine learning is a subset of artificial intelligence" in final_answer
        assert confidence == 0.9

    @patch("src.generation.ChatOpenAI")
    def test_parse_final_answer(self, mock_chat_openai, mock_llm):
        """Test parsing of final answer from LLM response."""
        mock_chat_openai.return_value = mock_llm

        generator = AnswerGenerator()

        answer_text = """
        Final Answer: Machine learning is a subset of artificial intelligence.
        Confidence: 0.9
        Evidence: Based on the retrieved documents.
        """

        final_answer, confidence = generator._parse_final_answer(answer_text)

        assert (
            final_answer == "Machine learning is a subset of artificial intelligence."
        )
        assert confidence == 0.9

    @patch("src.generation.ChatOpenAI")
    def test_calculate_overall_confidence(self, mock_chat_openai, mock_llm):
        """Test overall confidence calculation."""
        mock_chat_openai.return_value = mock_llm

        generator = AnswerGenerator()

        reasoning_steps = [
            ReasoningStep(
                step_number=1,
                question="Step 1?",
                retrieved_documents=[],
                reasoning="Test reasoning",
                confidence=0.8,
            ),
            ReasoningStep(
                step_number=2,
                question="Step 2?",
                retrieved_documents=[],
                reasoning="Test reasoning",
                confidence=0.9,
            ),
        ]

        overall_confidence = generator._calculate_overall_confidence(
            reasoning_steps, 0.85
        )

        # Should be weighted combination: (0.85 * 0.6) + (0.85 * 0.4) = 0.85
        expected_confidence = (0.85 * 0.6) + (0.85 * 0.4)
        assert abs(overall_confidence - expected_confidence) < 0.01

    @patch("src.generation.ChatOpenAI")
    def test_generate_multi_hop_answer(
        self, mock_chat_openai, mock_llm, sample_retrieval_results
    ):
        """Test complete multi-hop answer generation."""
        mock_chat_openai.return_value = mock_llm

        # Mock responses for different stages
        reasoning_response = AIMessage(
            content="""
        Step 1: What is machine learning?
        Reasoning: Based on the documents.
        Answer: Machine learning is a subset of AI.
        Confidence: 0.9
        """
        )

        final_answer_response = AIMessage(
            content="""
        Final Answer: Machine learning is a subset of artificial intelligence.
        Confidence: 0.9
        Evidence: The documents clearly explain this concept.
        """
        )

        mock_llm.invoke.side_effect = [reasoning_response, final_answer_response]

        generator = AnswerGenerator()

        query = MultiHopQuery(original_question="What is machine learning?")
        response = generator.generate_multi_hop_answer(query, sample_retrieval_results)

        assert isinstance(response, HANRAGResponse)
        assert response.answer is not None
        assert response.confidence > 0
        assert len(response.reasoning_chain) > 0
        assert len(response.retrieved_documents) > 0
        assert response.processing_time > 0


class TestNoiseResistantGenerator:
    """Test NoiseResistantGenerator class."""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing."""
        mock_llm = Mock(spec=ChatOpenAI)
        mock_llm.invoke.return_value = AIMessage(content="Test response")
        return mock_llm

    @pytest.fixture
    def sample_retrieval_results(self):
        """Sample retrieval results for testing."""
        return [
            RetrievalResult(
                document_id="doc1",
                content="High quality document about machine learning.",
                score=0.9,
                metadata={"source": "test"},
            ),
            RetrievalResult(
                document_id="doc2",
                content="Another good document about AI.",
                score=0.8,
                metadata={"source": "test"},
            ),
        ]

    @patch("src.generation.ChatOpenAI")
    def test_noise_resistant_generator_initialization(self, mock_chat_openai, mock_llm):
        """Test NoiseResistantGenerator initialization."""
        mock_chat_openai.return_value = mock_llm

        generator = NoiseResistantGenerator()

        assert generator.noise_threshold is not None
        assert hasattr(generator, "_clarify_noisy_query")
        assert hasattr(generator, "_filter_noisy_documents")
        assert hasattr(generator, "_detect_answer_noise")

    @patch("src.generation.ChatOpenAI")
    def test_clarify_noisy_query(self, mock_chat_openai, mock_llm):
        """Test query clarification for noisy queries."""
        mock_chat_openai.return_value = mock_llm

        # Mock LLM response for clarification
        clarification_response = AIMessage(
            content="What is machine learning and how does it enable computers to learn from data?"
        )
        mock_llm.invoke.return_value = clarification_response

        generator = NoiseResistantGenerator()

        original_question = "What is that thing about computers learning?"
        query = MultiHopQuery(original_question=original_question)
        clarified_query = generator._clarify_noisy_query(query)

        assert clarified_query.original_question != original_question
        assert "machine learning" in clarified_query.original_question.lower()

    @patch("src.generation.ChatOpenAI")
    def test_filter_noisy_documents(self, mock_chat_openai, mock_llm):
        """Test filtering of noisy documents."""
        mock_chat_openai.return_value = mock_llm

        generator = NoiseResistantGenerator()

        # Create documents with different quality levels
        documents = [
            RetrievalResult(
                document_id="doc1",
                content="High quality document with detailed information about machine learning algorithms and their applications.",
                score=0.9,
            ),
            RetrievalResult(
                document_id="doc2", content="Short", score=0.8  # Too short
            ),
            RetrievalResult(
                document_id="doc3",
                content="This is a placeholder text for testing purposes.",  # Contains noise
                score=0.7,
            ),
            RetrievalResult(
                document_id="doc4",
                content="Another comprehensive document about artificial intelligence and its various subfields.",
                score=0.6,
            ),
        ]

        filtered_docs = generator._filter_noisy_documents(documents)

        # Should filter out short and noisy documents
        assert len(filtered_docs) < len(documents)
        assert all(len(doc.content) > 50 for doc in filtered_docs)
        assert all("placeholder" not in doc.content.lower() for doc in filtered_docs)

    @patch("src.generation.ChatOpenAI")
    def test_detect_answer_noise(
        self, mock_chat_openai, mock_llm, sample_retrieval_results
    ):
        """Test answer noise detection."""
        mock_chat_openai.return_value = mock_llm

        generator = NoiseResistantGenerator()

        # Test with noisy answer
        noisy_answer = (
            "I don't know the answer to this question. I'm not sure about this."
        )
        noise_result = generator._detect_answer_noise(
            "What is machine learning?", noisy_answer, sample_retrieval_results
        )

        assert isinstance(noise_result, NoiseDetectionResult)
        assert noise_result.is_noisy is True
        assert noise_result.noise_score > 0

        # Test with clean answer
        clean_answer = "Machine learning is a subset of artificial intelligence that enables computers to learn from data."
        noise_result = generator._detect_answer_noise(
            "What is machine learning?", clean_answer, sample_retrieval_results
        )

        assert noise_result.is_noisy is False
        assert noise_result.noise_score < generator.noise_threshold

    @patch("src.generation.ChatOpenAI")
    def test_calculate_document_support(self, mock_chat_openai, mock_llm):
        """Test document support calculation."""
        mock_chat_openai.return_value = mock_llm

        generator = NoiseResistantGenerator()

        answer = "Machine learning is a subset of artificial intelligence"
        documents = [
            RetrievalResult(
                document_id="doc1",
                content="Machine learning is a subset of artificial intelligence that enables computers to learn.",
                score=0.9,
            ),
            RetrievalResult(
                document_id="doc2",
                content="Artificial intelligence includes machine learning and deep learning techniques.",
                score=0.8,
            ),
        ]

        support_score = generator._calculate_document_support(answer, documents)

        assert 0 <= support_score <= 1
        assert support_score > 0  # Should have some support

    @patch("src.generation.ChatOpenAI")
    def test_generate_with_noise_detection(
        self, mock_chat_openai, mock_llm, sample_retrieval_results
    ):
        """Test generation with noise detection."""
        mock_chat_openai.return_value = mock_llm

        # Mock responses for different stages
        reasoning_response = AIMessage(
            content="""
        Step 1: What is machine learning?
        Reasoning: Based on the documents.
        Answer: Machine learning is a subset of AI.
        Confidence: 0.9
        """
        )

        final_answer_response = AIMessage(
            content="""
        Final Answer: Machine learning is a subset of artificial intelligence.
        Confidence: 0.9
        Evidence: The documents clearly explain this concept.
        """
        )

        mock_llm.invoke.side_effect = [reasoning_response, final_answer_response]

        generator = NoiseResistantGenerator()

        query = MultiHopQuery(original_question="What is machine learning?")
        response = generator.generate_with_noise_detection(
            query, sample_retrieval_results, is_query_noisy=False
        )

        assert isinstance(response, HANRAGResponse)
        assert response.answer is not None
        assert response.confidence > 0
        assert "noise_detection" in response.metadata
        assert isinstance(response.metadata["noise_detection"], dict)
