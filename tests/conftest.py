"""
Pytest configuration and fixtures for HANRAG tests.
"""

import pytest
import os
from unittest.mock import Mock, patch
from langchain_core.documents import Document

# Set test environment variables
os.environ["OPENAI_API_KEY"] = "test-api-key"
os.environ["LANGSMITH_API_KEY"] = "test-langsmith-key"
os.environ["DEFAULT_MODEL"] = "gpt-3.5-turbo"
os.environ["EMBEDDING_MODEL"] = "text-embedding-ada-002"


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
            metadata={
                "id": "doc1",
                "type": "text",
                "domain": "technology",
                "source": "test",
            },
        ),
        Document(
            page_content="Deep learning is a subset of machine learning that uses neural networks with multiple layers to process complex patterns in data.",
            metadata={
                "id": "doc2",
                "type": "text",
                "domain": "technology",
                "source": "test",
            },
        ),
        Document(
            page_content="Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language.",
            metadata={
                "id": "doc3",
                "type": "text",
                "domain": "technology",
                "source": "test",
            },
        ),
        Document(
            page_content="Computer vision is a field of artificial intelligence that trains computers to interpret and understand visual information from the world.",
            metadata={
                "id": "doc4",
                "type": "text",
                "domain": "technology",
                "source": "test",
            },
        ),
        Document(
            page_content="Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward.",
            metadata={
                "id": "doc5",
                "type": "text",
                "domain": "technology",
                "source": "test",
            },
        ),
    ]


@pytest.fixture
def sample_retrieval_results():
    """Sample retrieval results for testing."""
    from src.models import RetrievalResult, DocumentType

    return [
        RetrievalResult(
            document_id="doc1",
            content="Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
            score=0.95,
            metadata={"source": "test", "domain": "technology"},
            document_type=DocumentType.TEXT,
        ),
        RetrievalResult(
            document_id="doc2",
            content="Deep learning uses neural networks with multiple layers to process complex patterns.",
            score=0.88,
            metadata={"source": "test", "domain": "technology"},
            document_type=DocumentType.TEXT,
        ),
        RetrievalResult(
            document_id="doc3",
            content="Natural language processing focuses on human-computer interaction through language.",
            score=0.82,
            metadata={"source": "test", "domain": "technology"},
            document_type=DocumentType.TEXT,
        ),
    ]


@pytest.fixture
def sample_reasoning_steps():
    """Sample reasoning steps for testing."""
    from src.models import ReasoningStep

    return [
        ReasoningStep(
            step_number=1,
            question="What is machine learning?",
            retrieved_documents=[],
            reasoning="Based on the retrieved documents, I need to understand what machine learning is.",
            intermediate_answer="Machine learning is a subset of artificial intelligence.",
            confidence=0.9,
        ),
        ReasoningStep(
            step_number=2,
            question="How does it relate to deep learning?",
            retrieved_documents=[],
            reasoning="Now I need to connect machine learning to deep learning concepts.",
            intermediate_answer="Deep learning is a subset of machine learning that uses neural networks.",
            confidence=0.85,
        ),
    ]


@pytest.fixture
def sample_multi_hop_query():
    """Sample multi-hop query for testing."""
    from src.models import MultiHopQuery

    return MultiHopQuery(
        original_question="What is the relationship between machine learning and deep learning?",
        reasoning_steps=[],
        final_answer=None,
        confidence=0.0,
        noise_detected=False,
        noise_level=0.0,
    )


@pytest.fixture
def mock_openai_llm():
    """Mock OpenAI LLM for testing."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import AIMessage

    mock_llm = Mock(spec=ChatOpenAI)
    mock_llm.invoke.return_value = AIMessage(content="Test response from LLM")
    return mock_llm


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing."""
    from langchain_openai.embeddings import OpenAIEmbeddings

    mock_emb = Mock(spec=OpenAIEmbeddings)
    mock_emb.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_emb.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock_emb


@pytest.fixture
def mock_vectorstore():
    """Mock vector store for testing."""
    from langchain_core.documents import Document

    mock_vs = Mock()
    mock_vs.similarity_search.return_value = [
        Document(page_content="Test document 1", metadata={"id": "doc1"}),
        Document(page_content="Test document 2", metadata={"id": "doc2"}),
    ]
    mock_vs.similarity_search_with_score.return_value = [
        (Document(page_content="Test document 1", metadata={"id": "doc1"}), 0.1),
        (Document(page_content="Test document 2", metadata={"id": "doc2"}), 0.2),
    ]
    return mock_vs


@pytest.fixture
def test_config():
    """Test configuration."""
    return {
        "top_k_documents": 3,
        "similarity_threshold": 0.7,
        "max_hops": 2,
        "noise_threshold": 0.3,
        "confidence_threshold": 0.8,
        "default_model": "gpt-3.5-turbo",
        "embedding_model": "text-embedding-ada-002",
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Mock external API calls
    with patch("openai.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Test response"))]
        )
        yield


@pytest.fixture
def mock_hanrag_system():
    """Mock HANRAG system for testing."""
    with patch("src.hanrag.NoiseResistantRetriever"), patch(
        "src.hanrag.NoiseResistantGenerator"
    ):
        from src.hanrag import HANRAGSystem

        return HANRAGSystem()


@pytest.fixture
def mock_langchain_integration(mock_hanrag_system):
    """Mock LangChain integration for testing."""
    with patch("src.langchain_integration.DocumentProcessor"), patch(
        "src.langchain_integration.LangChainRetriever"
    ):
        from src.langchain_integration import LangChainHANRAGIntegration

        return LangChainHANRAGIntegration(mock_hanrag_system)


@pytest.fixture
def mock_langgraph_workflow(mock_hanrag_system):
    """Mock LangGraph workflow for testing."""
    with patch("src.langgraph_integration.ChatOpenAI"):
        from src.langgraph_integration import HANRAGLangGraphWorkflow

        return HANRAGLangGraphWorkflow(mock_hanrag_system)


# Test data for different scenarios
@pytest.fixture
def noisy_queries():
    """Sample noisy queries for testing."""
    return [
        "What is that thing about computers learning stuff?",
        "Tell me about something related to AI or whatever",
        "How does that machine learning thing work?",
        "What about deep learning and neural networks and stuff?",
    ]


@pytest.fixture
def clear_queries():
    """Sample clear queries for testing."""
    return [
        "What is machine learning?",
        "How does deep learning work?",
        "What is the difference between supervised and unsupervised learning?",
        "Explain the concept of neural networks in artificial intelligence.",
    ]


@pytest.fixture
def multi_hop_queries():
    """Sample multi-hop queries for testing."""
    return [
        "What is the relationship between machine learning and deep learning?",
        "How do neural networks relate to artificial intelligence?",
        "What are the applications of natural language processing in machine learning?",
        "Explain how computer vision uses deep learning techniques.",
    ]


@pytest.fixture
def expected_answers():
    """Expected answers for test queries."""
    return {
        "What is machine learning?": "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        "How does deep learning work?": "Deep learning uses neural networks with multiple layers to process complex patterns in data.",
        "What is the relationship between machine learning and deep learning?": "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
    }


# Performance testing fixtures
@pytest.fixture
def large_document_set():
    """Large set of documents for performance testing."""
    documents = []
    for i in range(100):
        doc = Document(
            page_content=f"This is document {i} about various topics in artificial intelligence, machine learning, and computer science. "
            * 10,
            metadata={"id": f"doc_{i}", "type": "text", "domain": "technology"},
        )
        documents.append(doc)
    return documents


@pytest.fixture
def benchmark_queries():
    """Queries for benchmarking performance."""
    return [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are neural networks?",
        "Explain deep learning concepts",
        "What is natural language processing?",
        "How does computer vision work?",
        "What is reinforcement learning?",
        "Explain supervised learning",
        "What is unsupervised learning?",
        "How do transformers work in NLP?",
    ]
