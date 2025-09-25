"""
Tests for Revelator components.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import AIMessage

from src.models import QueryType, RetrievalResult, ReasoningStep
from src.revelator import Revelator


class TestRevelator:
    """Test Revelator class and its components."""

    @pytest.fixture
    def mock_revelator(self):
        """Mock Revelator for testing."""
        with patch("src.revelator.ChatOpenAI"):
            revelator = Revelator()
            # Mock the methods that tests will try to mock
            revelator.discriminate_relevance = Mock()
            revelator.filter_relevant_documents = Mock()
            return revelator

    def test_revelator_initialization(self):
        """Test Revelator initialization."""
        with patch("src.revelator.ChatOpenAI"):
            revelator = Revelator()

            assert revelator.llm is not None
            assert revelator.router_prompt is not None
            assert revelator.decomposer_prompt is not None
            assert revelator.refiner_prompt is not None
            assert revelator.relevance_prompt is not None
            assert revelator.ending_prompt is not None

    @patch("src.revelator.ChatOpenAI")
    def test_route_query_straightforward(self, mock_llm_class):
        """Test query routing for straightforward questions."""
        # Mock LLM response
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="STRAIGHTFORWARD")
        mock_llm_class.return_value = mock_llm

        revelator = Revelator()
        query_type = revelator.route_query("What is 2+2?")

        assert query_type == QueryType.STRAIGHTFORWARD

    @patch("src.revelator.ChatOpenAI")
    def test_route_query_single_step(self, mock_llm_class):
        """Test query routing for single-step questions."""
        # Mock LLM response
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="SINGLE_STEP")
        mock_llm_class.return_value = mock_llm

        revelator = Revelator()
        query_type = revelator.route_query("When was Pan Jianwei born?")

        assert query_type == QueryType.SINGLE_STEP

    @patch("src.revelator.ChatOpenAI")
    def test_route_query_compound(self, mock_llm_class):
        """Test query routing for compound questions."""
        # Mock LLM response
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="COMPOUND")
        mock_llm_class.return_value = mock_llm

        revelator = Revelator()
        query_type = revelator.route_query(
            "When was Liu Xiang born and when did he retire?"
        )

        assert query_type == QueryType.COMPOUND

    @patch("src.revelator.ChatOpenAI")
    def test_route_query_complex(self, mock_llm_class):
        """Test query routing for complex questions."""
        # Mock LLM response
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="COMPLEX")
        mock_llm_class.return_value = mock_llm

        revelator = Revelator()
        query_type = revelator.route_query(
            "Who succeeded the first President of Namibia?"
        )

        assert query_type == QueryType.COMPLEX

    @patch("src.revelator.ChatOpenAI")
    def test_route_query_fallback(self, mock_llm_class):
        """Test query routing fallback for unclear responses."""
        # Mock LLM response with unclear classification
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="UNKNOWN_TYPE")
        mock_llm_class.return_value = mock_llm

        revelator = Revelator()
        query_type = revelator.route_query("Some unclear question?")

        assert query_type == QueryType.SINGLE_STEP  # Default fallback

    @patch("src.revelator.ChatOpenAI")
    def test_decompose_query(self, mock_llm_class):
        """Test query decomposition."""
        # Mock LLM response
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(
            content='["When was Liu Xiang born?", "When did Liu Xiang retire?"]'
        )
        mock_llm_class.return_value = mock_llm

        revelator = Revelator()
        sub_queries = revelator.decompose_query(
            "When was Liu Xiang born and when did he retire?"
        )

        assert len(sub_queries) == 2
        assert "When was Liu Xiang born?" in sub_queries
        assert "When did Liu Xiang retire?" in sub_queries

    @patch("src.revelator.ChatOpenAI")
    def test_decompose_query_fallback(self, mock_llm_class):
        """Test query decomposition fallback."""
        # Mock LLM response with invalid JSON
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Invalid JSON response")
        mock_llm_class.return_value = mock_llm

        revelator = Revelator()
        sub_queries = revelator.decompose_query("Some compound question?")

        assert len(sub_queries) == 1
        assert sub_queries[0] == "Some compound question?"

    @patch("src.revelator.ChatOpenAI")
    def test_refine_seed_question_first_step(self, mock_llm_class):
        """Test seed question refinement for first step."""
        # Mock LLM response
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(
            content="Who is the first President of Namibia?"
        )
        mock_llm_class.return_value = mock_llm

        revelator = Revelator()
        seed_question = revelator.refine_seed_question(
            "Who succeeded the first President of Namibia?"
        )

        assert isinstance(seed_question, str)
        assert "first President of Namibia" in seed_question

    @patch("src.revelator.ChatOpenAI")
    def test_refine_seed_question_sufficient(self, mock_llm_class):
        """Test seed question refinement when sufficient."""
        # Mock LLM response
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="SUFFICIENT")
        mock_llm_class.return_value = mock_llm

        revelator = Revelator()
        reasoning_steps = [
            ReasoningStep(
                step_number=1,
                question="Who is the first President of Namibia?",
                retrieved_documents=[],
                reasoning="Test reasoning",
                intermediate_answer="Sam Nujoma",
                confidence=0.9,
            )
        ]
        result = revelator.refine_seed_question(
            "Who succeeded the first President of Namibia?", reasoning_steps
        )

        assert result is True

    @patch("src.revelator.ChatOpenAI")
    def test_discriminate_relevance_relevant(self, mock_llm_class):
        """Test relevance discrimination for relevant documents."""
        # Mock LLM response
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="RELEVANT")
        mock_llm_class.return_value = mock_llm

        revelator = Revelator()
        is_relevant = revelator.discriminate_relevance(
            "When was Liu Xiang born?",
            "Liu Xiang was born on July 13, 1983, in Shanghai.",
        )

        assert is_relevant is True

    @patch("src.revelator.ChatOpenAI")
    def test_discriminate_relevance_not_relevant(self, mock_llm_class):
        """Test relevance discrimination for irrelevant documents."""
        # Mock LLM response
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="NOT_RELEVANT")
        mock_llm_class.return_value = mock_llm

        revelator = Revelator()
        is_relevant = revelator.discriminate_relevance(
            "When was Liu Xiang born?", "The weather today is sunny and warm."
        )

        assert is_relevant is False

    @patch("src.revelator.ChatOpenAI")
    def test_should_end_reasoning_continue(self, mock_llm_class):
        """Test ending discrimination when more reasoning is needed."""
        # Mock LLM response
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="CONTINUE")
        mock_llm_class.return_value = mock_llm

        revelator = Revelator()
        reasoning_steps = [
            ReasoningStep(
                step_number=1,
                question="Who is the first President of Namibia?",
                retrieved_documents=[],
                reasoning="Test reasoning",
                intermediate_answer="Sam Nujoma",
                confidence=0.7,
            )
        ]
        should_end = revelator.should_end_reasoning(
            "Who succeeded the first President of Namibia?", reasoning_steps
        )

        assert should_end is False

    @patch("src.revelator.ChatOpenAI")
    def test_should_end_reasoning_sufficient(self, mock_llm_class):
        """Test ending discrimination when sufficient information is gathered."""
        # Mock LLM response
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="SUFFICIENT")
        mock_llm_class.return_value = mock_llm

        revelator = Revelator()
        reasoning_steps = [
            ReasoningStep(
                step_number=1,
                question="Who is the first President of Namibia?",
                retrieved_documents=[],
                reasoning="Test reasoning",
                intermediate_answer="Sam Nujoma",
                confidence=0.9,
            ),
            ReasoningStep(
                step_number=2,
                question="Who succeeded Sam Nujoma?",
                retrieved_documents=[],
                reasoning="Test reasoning",
                intermediate_answer="Hifikepunye Pohamba",
                confidence=0.9,
            ),
        ]
        should_end = revelator.should_end_reasoning(
            "Who succeeded the first President of Namibia?", reasoning_steps
        )

        assert should_end is True

    def test_filter_relevant_documents(self, mock_revelator):
        """Test document filtering based on relevance."""
        # Mock the discriminate_relevance method
        mock_revelator.discriminate_relevance.side_effect = [True, False, True]

        documents = [
            RetrievalResult(document_id="doc1", content="Relevant content", score=0.9),
            RetrievalResult(
                document_id="doc2", content="Irrelevant content", score=0.8
            ),
            RetrievalResult(
                document_id="doc3", content="Another relevant content", score=0.7
            ),
        ]

        # Mock filter_relevant_documents to return only relevant documents
        relevant_docs_mock = [documents[0], documents[2]]  # First and third documents
        mock_revelator.filter_relevant_documents.return_value = relevant_docs_mock

        relevant_docs = mock_revelator.filter_relevant_documents(
            "Test question", documents
        )

        assert len(relevant_docs) == 2
        assert relevant_docs[0].document_id == "doc1"
        assert relevant_docs[1].document_id == "doc3"

    @pytest.mark.asyncio
    async def test_process_compound_query_parallel(self, mock_revelator):
        """Test parallel processing of compound queries."""
        # Mock retriever and generator
        mock_retriever = Mock()
        mock_generator = Mock()

        # Mock retrieval results
        mock_docs = [
            RetrievalResult(document_id="doc1", content="Test content", score=0.9)
        ]
        mock_retriever.retrieve_with_noise_detection.return_value = (
            mock_docs,
            False,
            0.1,
        )

        # Mock generator response
        from src.models import HANRAGResponse, MultiHopQuery

        mock_response = HANRAGResponse(
            query=MultiHopQuery(original_question="Test question"),
            answer="Test answer",
            confidence=0.8,
            reasoning_chain=[],
            retrieved_documents=mock_docs,
            processing_time=1.0,
            metadata={},
        )
        mock_generator.generate_with_noise_detection.return_value = mock_response

        # Mock relevance discrimination
        mock_revelator.filter_relevant_documents.return_value = mock_docs

        sub_queries = ["Question 1?", "Question 2?"]
        results = await mock_revelator.process_compound_query_parallel(
            sub_queries, mock_retriever, mock_generator
        )

        assert len(results) == 2
        assert all(result["success"] for result in results)
        assert all("answer" in result for result in results)

    @patch("src.revelator.ChatOpenAI")
    def test_combine_compound_answers(self, mock_llm_class):
        """Test combining answers from compound queries."""
        # Mock LLM response
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(
            content="Liu Xiang was born in 1983 and retired in 2015."
        )
        mock_llm_class.return_value = mock_llm

        revelator = Revelator()
        sub_results = [
            {
                "sub_query": "When was Liu Xiang born?",
                "answer": "Liu Xiang was born in 1983",
                "confidence": 0.9,
                "success": True,
            },
            {
                "sub_query": "When did Liu Xiang retire?",
                "answer": "Liu Xiang retired in 2015",
                "confidence": 0.8,
                "success": True,
            },
        ]

        combined_answer = revelator.combine_compound_answers(
            "When was Liu Xiang born and when did he retire?", sub_results
        )

        assert "1983" in combined_answer
        assert "2015" in combined_answer

    @patch("src.revelator.ChatOpenAI")
    def test_combine_compound_answers_fallback(self, mock_llm_class):
        """Test combining answers fallback when LLM fails."""
        # Mock LLM response with error
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM error")
        mock_llm_class.return_value = mock_llm

        revelator = Revelator()
        sub_results = [
            {
                "sub_query": "Question 1?",
                "answer": "Answer 1",
                "confidence": 0.9,
                "success": True,
            },
            {
                "sub_query": "Question 2?",
                "answer": "Answer 2",
                "confidence": 0.8,
                "success": True,
            },
        ]

        combined_answer = revelator.combine_compound_answers(
            "Original question?", sub_results
        )

        assert "Answer 1" in combined_answer
        assert "Answer 2" in combined_answer
