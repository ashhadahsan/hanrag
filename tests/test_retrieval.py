"""
Tests for HANRAG retrieval components.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings

from src.models import RetrievalResult, DocumentType, HeuristicRule
from src.retrieval import HeuristicRetriever, NoiseResistantRetriever


class TestHeuristicRetriever:
    """Test HeuristicRetriever class."""

    @pytest.fixture
    def mock_embeddings(self):
        """Mock embeddings for testing."""
        return Mock(spec=OpenAIEmbeddings)

    @pytest.fixture
    def mock_vectorstore(self):
        """Mock vector store for testing."""
        mock_vs = Mock()
        mock_vs.similarity_search_with_score.return_value = [
            (Document(page_content="Test document 1", metadata={"id": "doc1"}), 0.1),
            (Document(page_content="Test document 2", metadata={"id": "doc2"}), 0.2),
        ]
        return mock_vs

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            Document(
                page_content="This is a test document about machine learning.",
                metadata={"id": "doc1", "type": "text", "domain": "technology"},
            ),
            Document(
                page_content="Another document about artificial intelligence.",
                metadata={"id": "doc2", "type": "text", "domain": "technology"},
            ),
            Document(
                page_content="A document about recent research in AI.",
                metadata={
                    "id": "doc3",
                    "type": "scientific_paper",
                    "domain": "science",
                    "recent": True,
                },
            ),
        ]

    @patch("src.retrieval.OpenAIEmbeddings")
    def test_heuristic_retriever_initialization(
        self, mock_embeddings_class, mock_embeddings
    ):
        """Test HeuristicRetriever initialization."""
        mock_embeddings_class.return_value = mock_embeddings

        retriever = HeuristicRetriever()

        assert retriever.embedding_model is not None
        assert len(retriever.heuristic_rules) > 0
        assert retriever.vectorstore is None

    @patch("src.retrieval.OpenAIEmbeddings")
    @patch("src.retrieval.FAISS")
    def test_add_documents(
        self, mock_faiss, mock_embeddings_class, mock_embeddings, sample_documents
    ):
        """Test adding documents to retriever."""
        mock_embeddings_class.return_value = mock_embeddings
        mock_faiss.from_documents.return_value = Mock()

        retriever = HeuristicRetriever()
        retriever.add_documents(sample_documents)

        mock_faiss.from_documents.assert_called_once_with(
            sample_documents, mock_embeddings
        )
        assert retriever.vectorstore is not None

    @patch("src.retrieval.OpenAIEmbeddings")
    def test_retrieve_documents_no_vectorstore(
        self, mock_embeddings_class, mock_embeddings
    ):
        """Test retrieve_documents when no vectorstore exists."""
        mock_embeddings_class.return_value = mock_embeddings

        retriever = HeuristicRetriever()

        with pytest.raises(ValueError, match="No documents have been added"):
            retriever.retrieve_documents("test query")

    @patch("src.retrieval.OpenAIEmbeddings")
    def test_retrieve_documents_with_heuristics(
        self, mock_embeddings_class, mock_embeddings, mock_vectorstore, sample_documents
    ):
        """Test document retrieval with heuristic rules applied."""
        mock_embeddings_class.return_value = mock_embeddings

        retriever = HeuristicRetriever()
        retriever.vectorstore = mock_vectorstore

        results = retriever.retrieve_documents("machine learning", k=2)

        assert len(results) == 2
        assert all(isinstance(result, RetrievalResult) for result in results)
        assert all(result.document_id in ["doc1", "doc2"] for result in results)
        assert all(result.score > 0 for result in results)

    def test_has_entity_overlap(self):
        """Test entity overlap detection."""
        retriever = HeuristicRetriever()

        # Test with overlap
        assert (
            retriever._has_entity_overlap(
                "machine learning", "This is about machine learning algorithms"
            )
            is True
        )

        # Test without overlap
        assert (
            retriever._has_entity_overlap("cooking", "This is about machine learning")
            is False
        )

    def test_has_temporal_marker(self):
        """Test temporal marker detection."""
        retriever = HeuristicRetriever()

        # Test with temporal markers
        assert retriever._has_temporal_marker("What is the latest research?") is True
        assert retriever._has_temporal_marker("Recent developments in AI") is True

        # Test without temporal markers
        assert retriever._has_temporal_marker("What is machine learning?") is False

    def test_is_recent_document(self):
        """Test recent document detection."""
        retriever = HeuristicRetriever()

        # Test recent document
        recent_metadata = {"recent": True}
        assert retriever._is_recent_document(recent_metadata) is True

        # Test non-recent document
        old_metadata = {"recent": False}
        assert retriever._is_recent_document(old_metadata) is False

    def test_domain_matches(self):
        """Test domain matching."""
        retriever = HeuristicRetriever()

        # Test matching domains
        assert (
            retriever._domain_matches("machine learning", {"domain": "technology"})
            is True
        )
        assert retriever._domain_matches("AI research", {"domain": "science"}) is True

        # Test non-matching domains
        assert retriever._domain_matches("cooking", {"domain": "technology"}) is False

    def test_extract_domain_from_query(self):
        """Test domain extraction from query."""
        retriever = HeuristicRetriever()

        # Test science domain
        assert (
            retriever._extract_domain_from_query("What is the latest research?")
            == "science"
        )

        # Test technology domain
        assert (
            retriever._extract_domain_from_query("How does machine learning work?")
            == "technology"
        )

        # Test general domain
        assert retriever._extract_domain_from_query("What is the weather?") == "general"


class TestNoiseResistantRetriever:
    """Test NoiseResistantRetriever class."""

    @pytest.fixture
    def mock_embeddings(self):
        """Mock embeddings for testing."""
        return Mock(spec=OpenAIEmbeddings)

    @pytest.fixture
    def mock_vectorstore(self):
        """Mock vector store for testing."""
        mock_vs = Mock()
        mock_vs.similarity_search_with_score.return_value = [
            (
                Document(page_content="High quality document", metadata={"id": "doc1"}),
                0.1,
            ),
            (
                Document(page_content="Another good document", metadata={"id": "doc2"}),
                0.2,
            ),
        ]
        return mock_vs

    @patch("src.retrieval.OpenAIEmbeddings")
    def test_noise_resistant_retriever_initialization(
        self, mock_embeddings_class, mock_embeddings
    ):
        """Test NoiseResistantRetriever initialization."""
        mock_embeddings_class.return_value = mock_embeddings

        retriever = NoiseResistantRetriever()

        assert retriever.noise_threshold is not None
        assert hasattr(retriever, "_detect_query_noise")
        assert hasattr(retriever, "_apply_noise_resistance")

    @patch("src.retrieval.OpenAIEmbeddings")
    def test_detect_query_noise(self, mock_embeddings_class, mock_embeddings):
        """Test query noise detection."""
        mock_embeddings_class.return_value = mock_embeddings

        retriever = NoiseResistantRetriever()

        # Test noisy query
        is_noisy, noise_score = retriever._detect_query_noise(
            "What is that thing about something?"
        )
        assert is_noisy is True
        assert noise_score > 0

        # Test clear query
        is_noisy, noise_score = retriever._detect_query_noise(
            "What is machine learning?"
        )
        assert is_noisy is False
        assert noise_score >= 0

    @patch("src.retrieval.OpenAIEmbeddings")
    def test_expand_query(self, mock_embeddings_class, mock_embeddings):
        """Test query expansion."""
        mock_embeddings_class.return_value = mock_embeddings

        retriever = NoiseResistantRetriever()

        # Test query expansion
        expanded = retriever._expand_query("This is a good example")
        assert "good" in expanded
        assert "excellent" in expanded or "great" in expanded

    @patch("src.retrieval.OpenAIEmbeddings")
    def test_combine_and_rerank(self, mock_embeddings_class, mock_embeddings):
        """Test document combination and re-ranking."""
        mock_embeddings_class.return_value = mock_embeddings

        retriever = NoiseResistantRetriever()

        # Create test results
        results1 = [
            RetrievalResult(document_id="doc1", content="Content 1", score=0.8),
            RetrievalResult(document_id="doc2", content="Content 2", score=0.6),
        ]

        results2 = [
            RetrievalResult(document_id="doc2", content="Content 2 updated", score=0.9),
            RetrievalResult(document_id="doc3", content="Content 3", score=0.7),
        ]

        combined = retriever._combine_and_rerank(results1, results2)

        assert len(combined) == 3  # Should have 3 unique documents
        assert combined[0].document_id == "doc2"  # Should be highest scoring
        assert combined[0].score == 0.9  # Should have the higher score

    @patch("src.retrieval.OpenAIEmbeddings")
    def test_retrieve_with_noise_detection(
        self, mock_embeddings_class, mock_embeddings, mock_vectorstore
    ):
        """Test retrieval with noise detection."""
        mock_embeddings_class.return_value = mock_embeddings

        retriever = NoiseResistantRetriever()
        retriever.vectorstore = mock_vectorstore

        # Test with clear query
        results, is_noisy, noise_score = retriever.retrieve_with_noise_detection(
            "What is machine learning?"
        )

        assert len(results) > 0
        assert is_noisy is False
        assert noise_score >= 0
        assert all(isinstance(result, RetrievalResult) for result in results)

    @patch("src.retrieval.OpenAIEmbeddings")
    def test_filter_noisy_documents(self, mock_embeddings_class, mock_embeddings):
        """Test filtering of noisy documents."""
        mock_embeddings_class.return_value = mock_embeddings

        retriever = NoiseResistantRetriever()

        # Create test documents with different quality levels
        documents = [
            RetrievalResult(
                document_id="doc1",
                content="High quality content with good information",
                score=0.9,
            ),
            RetrievalResult(
                document_id="doc2", content="Short", score=0.8
            ),  # Too short
            RetrievalResult(
                document_id="doc3", content="This is a placeholder text", score=0.7
            ),  # Contains noise
            RetrievalResult(
                document_id="doc4",
                content="Another high quality document with detailed information",
                score=0.6,
            ),
        ]

        filtered = retriever._filter_noisy_documents(documents)

        # Should filter out short and noisy documents
        assert len(filtered) < len(documents)
        assert all(len(doc.content) > 50 for doc in filtered)
        assert all("placeholder" not in doc.content.lower() for doc in filtered)

    def test_contains_noise_indicators(self):
        """Test noise indicator detection in content."""
        retriever = NoiseResistantRetriever()

        # Test content with noise indicators
        assert retriever._contains_noise_indicators("This is an error message") is True
        assert (
            retriever._contains_noise_indicators("Lorem ipsum dolor sit amet") is True
        )
        assert retriever._contains_noise_indicators("This is test content") is True

        # Test clean content
        assert (
            retriever._contains_noise_indicators(
                "This is a well-written article about machine learning"
            )
            is False
        )
        assert (
            retriever._contains_noise_indicators(
                "The research shows interesting results"
            )
            is False
        )

    def test_add_documents_no_vectorstore(self, mock_embeddings):
        """Test adding documents when no vectorstore exists."""
        with patch("src.retrieval.OpenAIEmbeddings") as mock_embeddings_class:
            mock_embeddings_class.return_value = mock_embeddings

            retriever = NoiseResistantRetriever()
            documents = [
                Document(page_content="Test document", metadata={"id": "doc1"})
            ]

            with patch("src.retrieval.FAISS") as mock_faiss:
                mock_vectorstore = Mock()
                mock_faiss.from_documents.return_value = mock_vectorstore

                retriever.add_documents(documents)

                mock_faiss.from_documents.assert_called_once_with(
                    documents, mock_embeddings
                )
                assert retriever.vectorstore == mock_vectorstore

    def test_add_documents_existing_vectorstore(self, mock_embeddings):
        """Test adding documents when vectorstore already exists."""
        with patch("src.retrieval.OpenAIEmbeddings") as mock_embeddings_class:
            mock_embeddings_class.return_value = mock_embeddings

            retriever = NoiseResistantRetriever()
            mock_vectorstore = Mock()
            retriever.vectorstore = mock_vectorstore

            documents = [
                Document(page_content="Test document", metadata={"id": "doc1"})
            ]

            retriever.add_documents(documents)

            mock_vectorstore.add_documents.assert_called_once_with(documents)

    def test_retrieve_documents_no_vectorstore(self, mock_embeddings):
        """Test retrieve_documents when no vectorstore exists."""
        with patch("src.retrieval.OpenAIEmbeddings") as mock_embeddings_class:
            mock_embeddings_class.return_value = mock_embeddings

            retriever = NoiseResistantRetriever()

            with pytest.raises(
                ValueError, match="No documents have been added to the retriever"
            ):
                retriever.retrieve_documents("test query")

    def test_retrieve_documents_with_heuristics(
        self, mock_embeddings, mock_vectorstore
    ):
        """Test retrieve_documents with heuristics enabled."""
        with patch("src.retrieval.OpenAIEmbeddings") as mock_embeddings_class:
            mock_embeddings_class.return_value = mock_embeddings

            retriever = NoiseResistantRetriever()
            retriever.vectorstore = mock_vectorstore

            with patch.object(retriever, "_apply_heuristic_rules") as mock_heuristics:
                mock_heuristics.return_value = [
                    (Document(page_content="Test doc 1", metadata={"id": "doc1"}), 0.1),
                    (Document(page_content="Test doc 2", metadata={"id": "doc2"}), 0.2),
                ]

                results = retriever.retrieve_documents(
                    "test query", k=2, apply_heuristics=True
                )

                assert len(results) == 2
                assert all(isinstance(r, RetrievalResult) for r in results)
                mock_heuristics.assert_called_once()

    def test_retrieve_documents_without_heuristics(
        self, mock_embeddings, mock_vectorstore
    ):
        """Test retrieve_documents with heuristics disabled."""
        with patch("src.retrieval.OpenAIEmbeddings") as mock_embeddings_class:
            mock_embeddings_class.return_value = mock_embeddings

            retriever = NoiseResistantRetriever()
            retriever.vectorstore = mock_vectorstore

            with patch.object(retriever, "_apply_heuristic_rules") as mock_heuristics:
                results = retriever.retrieve_documents(
                    "test query", k=2, apply_heuristics=False
                )

                assert len(results) == 2
                assert all(isinstance(r, RetrievalResult) for r in results)
                mock_heuristics.assert_not_called()

    def test_retrieve_with_noise_detection_clean_query(
        self, mock_embeddings, mock_vectorstore
    ):
        """Test retrieve_with_noise_detection with clean query."""
        with patch("src.retrieval.OpenAIEmbeddings") as mock_embeddings_class:
            mock_embeddings_class.return_value = mock_embeddings

            retriever = NoiseResistantRetriever()
            retriever.vectorstore = mock_vectorstore

            with patch.object(retriever, "_detect_query_noise") as mock_detect:
                mock_detect.return_value = (False, 0.05)  # Clean query

                docs, is_noisy, noise_score = retriever.retrieve_with_noise_detection(
                    "clean query"
                )

                assert is_noisy is False
                assert noise_score == 0.05
                assert len(docs) == 2  # From mock_vectorstore

    def test_retrieve_with_noise_detection_noisy_query(
        self, mock_embeddings, mock_vectorstore
    ):
        """Test retrieve_with_noise_detection with noisy query."""
        with patch("src.retrieval.OpenAIEmbeddings") as mock_embeddings_class:
            mock_embeddings_class.return_value = mock_embeddings

            retriever = NoiseResistantRetriever()
            retriever.vectorstore = mock_vectorstore

            with patch.object(retriever, "_detect_query_noise") as mock_detect:
                mock_detect.return_value = (True, 0.8)  # Noisy query

                docs, is_noisy, noise_score = retriever.retrieve_with_noise_detection(
                    "noisy query"
                )

                assert is_noisy is True
                assert noise_score == 0.8
                assert len(docs) == 2  # From mock_vectorstore
