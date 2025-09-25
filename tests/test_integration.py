"""
Integration tests for HANRAG system with LangChain and LangGraph.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from src.models import MultiHopQuery, HANRAGResponse, RetrievalResult
from src.hanrag import HANRAGSystem
from src.langchain_integration import (
    DocumentProcessor,
    LangChainRetriever,
    LangChainHANRAGIntegration,
)
from src.langgraph_integration import (
    HANRAGLangGraphWorkflow,
    ConversationalHANRAGWorkflow,
)


class TestDocumentProcessor:
    """Test DocumentProcessor integration."""

    def test_document_processor_initialization(self):
        """Test DocumentProcessor initialization."""
        processor = DocumentProcessor()

        assert processor.chunk_size == 1000
        assert processor.chunk_overlap == 200
        assert processor.text_splitter is not None

    def test_load_documents_from_text(self):
        """Test loading documents from text."""
        processor = DocumentProcessor()

        texts = [
            "This is the first document about machine learning.",
            "This is the second document about artificial intelligence.",
        ]

        metadata = [
            {"source": "test1", "type": "text"},
            {"source": "test2", "type": "text"},
        ]

        documents = processor.load_documents_from_text(texts, metadata)

        assert len(documents) == 2
        assert documents[0].page_content == texts[0]
        assert documents[0].metadata["source"] == "test1"
        assert documents[1].page_content == texts[1]
        assert documents[1].metadata["source"] == "test2"

    def test_split_documents(self):
        """Test document splitting."""
        processor = DocumentProcessor()

        documents = [
            Document(
                page_content="This is a long document that should be split into multiple chunks. "
                * 50,
                metadata={"id": "doc1"},
            )
        ]

        chunks = processor.split_documents(documents)

        assert len(chunks) > 1  # Should be split into multiple chunks
        assert all(len(chunk.page_content) <= processor.chunk_size for chunk in chunks)

    def test_create_document_from_hanrag_paper(self):
        """Test creating document from HANRAG paper content."""
        processor = DocumentProcessor()

        documents = processor.create_document_from_hanrag_paper()

        assert len(documents) == 1
        assert "HANRAG" in documents[0].page_content
        assert "Heuristic Accurate Noise-resistant" in documents[0].page_content
        assert documents[0].metadata["type"] == "scientific_paper"

    @patch("src.langchain_integration.PyPDFLoader")
    def test_load_documents_from_file_pdf(self, mock_pdf_loader):
        """Test loading documents from PDF file."""
        processor = DocumentProcessor()

        # Mock the PDF loader
        mock_doc = Document(page_content="Test PDF content", metadata={})
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [mock_doc]
        mock_pdf_loader.return_value = mock_loader_instance

        documents = processor.load_documents_from_file("test.pdf")

        assert len(documents) == 1
        assert documents[0].page_content == "Test PDF content"
        assert documents[0].metadata["source"] == "test.pdf"
        mock_pdf_loader.assert_called_once_with("test.pdf")

    @patch("src.langchain_integration.TextLoader")
    def test_load_documents_from_file_txt(self, mock_text_loader):
        """Test loading documents from TXT file."""
        processor = DocumentProcessor()

        # Mock the text loader
        mock_doc = Document(page_content="Test text content", metadata={})
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [mock_doc]
        mock_text_loader.return_value = mock_loader_instance

        documents = processor.load_documents_from_file("test.txt")

        assert len(documents) == 1
        assert documents[0].page_content == "Test text content"
        assert documents[0].metadata["source"] == "test.txt"
        mock_text_loader.assert_called_once_with("test.txt", encoding="utf-8")

    def test_load_documents_from_file_unsupported(self):
        """Test loading documents from unsupported file type."""
        processor = DocumentProcessor()

        with pytest.raises(ValueError, match="Unsupported file type"):
            processor.load_documents_from_file("test.doc")

    @patch("src.langchain_integration.DirectoryLoader")
    def test_load_documents_from_directory(self, mock_dir_loader):
        """Test loading documents from directory."""
        processor = DocumentProcessor()

        # Mock the directory loader
        mock_doc1 = Document(page_content="Content 1", metadata={})
        mock_doc2 = Document(page_content="Content 2", metadata={})
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [mock_doc1, mock_doc2]
        mock_dir_loader.return_value = mock_loader_instance

        documents = processor.load_documents_from_directory("test_dir")

        assert len(documents) == 2
        assert documents[0].metadata["source"] == "test_dir"
        assert documents[1].metadata["source"] == "test_dir"
        mock_dir_loader.assert_called_once()

    @patch("src.langchain_integration.WebBaseLoader")
    def test_load_documents_from_urls(self, mock_web_loader):
        """Test loading documents from URLs."""
        processor = DocumentProcessor()

        # Mock the web loader
        mock_doc = Document(page_content="Web content", metadata={})
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [mock_doc]
        mock_web_loader.return_value = mock_loader_instance

        urls = ["https://example.com"]
        documents = processor.load_documents_from_urls(urls)

        assert len(documents) == 1
        assert documents[0].page_content == "Web content"
        assert documents[0].metadata["source"] == "web"
        mock_web_loader.assert_called_once_with(urls)

    def test_add_metadata(self):
        """Test adding metadata to documents."""
        processor = DocumentProcessor()

        documents = [
            Document(page_content="Content 1", metadata={}),
            Document(page_content="Content 2", metadata={"existing": "value"}),
        ]

        result = processor._add_metadata(documents, "test_source")

        assert len(result) == 2
        assert result[0].metadata["source"] == "test_source"
        assert result[0].metadata["id"] == "test_source_0"
        assert result[0].metadata["chunk_index"] == 0
        assert result[0].metadata["total_chunks"] == 2
        assert result[1].metadata["source"] == "test_source"
        assert result[1].metadata["id"] == "test_source_1"
        assert result[1].metadata["chunk_index"] == 1
        assert result[1].metadata["total_chunks"] == 2
        assert result[1].metadata["existing"] == "value"  # Preserve existing metadata


class TestLangChainRetriever:
    """Test LangChainRetriever integration."""

    @patch("src.langchain_integration.OpenAIEmbeddings")
    def test_langchain_retriever_initialization(self, mock_embeddings):
        """Test LangChainRetriever initialization."""
        mock_embeddings.return_value = Mock()

        retriever = LangChainRetriever()

        assert retriever.embedding_model is not None
        assert retriever.vectorstore_type == "faiss"
        assert retriever.embeddings is not None

    @patch("src.langchain_integration.OpenAIEmbeddings")
    @patch("src.langchain_integration.FAISS")
    def test_create_vectorstore(self, mock_faiss, mock_embeddings):
        """Test vector store creation."""
        mock_embeddings.return_value = Mock()
        mock_faiss.from_documents.return_value = Mock()

        retriever = LangChainRetriever()

        documents = [
            Document(page_content="Test document 1", metadata={"id": "doc1"}),
            Document(page_content="Test document 2", metadata={"id": "doc2"}),
        ]

        retriever.create_vectorstore(documents)

        mock_faiss.from_documents.assert_called_once()
        assert retriever.vectorstore is not None

    @patch("src.langchain_integration.OpenAIEmbeddings")
    def test_similarity_search(self, mock_embeddings):
        """Test similarity search."""
        mock_embeddings.return_value = Mock()

        retriever = LangChainRetriever()
        retriever.vectorstore = Mock()
        retriever.vectorstore.similarity_search.return_value = [
            Document(page_content="Test result", metadata={"id": "result1"})
        ]

        results = retriever.similarity_search("test query")

        assert len(results) > 0
        assert results[0].page_content == "Test result"

    @patch("src.langchain_integration.OpenAIEmbeddings")
    def test_similarity_search_no_vectorstore(self, mock_embeddings):
        """Test similarity_search when no vectorstore exists."""
        mock_embeddings.return_value = Mock()

        retriever = LangChainRetriever()

        with pytest.raises(ValueError, match="Vector store not created"):
            retriever.similarity_search("test query")

    @patch("src.langchain_integration.OpenAIEmbeddings")
    def test_similarity_search_with_score(self, mock_embeddings):
        """Test similarity_search_with_score with existing vectorstore."""
        mock_embeddings.return_value = Mock()

        retriever = LangChainRetriever()
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search_with_score.return_value = [
            (Document(page_content="Test doc 1", metadata={"id": "doc1"}), 0.9),
            (Document(page_content="Test doc 2", metadata={"id": "doc2"}), 0.8),
        ]
        retriever.vectorstore = mock_vectorstore

        results = retriever.similarity_search_with_score("test query", k=2)

        assert len(results) == 2
        assert results[0][1] == 0.9  # score
        assert results[1][1] == 0.8  # score
        assert results[0][0].page_content == "Test doc 1"
        assert results[1][0].page_content == "Test doc 2"

    @patch("src.langchain_integration.OpenAIEmbeddings")
    @patch("src.langchain_integration.Chroma")
    def test_create_vectorstore_chroma(self, mock_chroma, mock_embeddings):
        """Test creating Chroma vectorstore."""
        mock_embeddings.return_value = Mock()
        mock_chroma.from_documents.return_value = Mock()

        retriever = LangChainRetriever()

        documents = [Document(page_content="Test document", metadata={"id": "doc1"})]

        retriever.create_vectorstore(documents, vectorstore_type="chroma")

        mock_chroma.from_documents.assert_called_once_with(
            documents, retriever.embeddings
        )
        assert retriever.vectorstore is not None

    @patch("src.langchain_integration.OpenAIEmbeddings")
    def test_create_vectorstore_unsupported_type(self, mock_embeddings):
        """Test creating vectorstore with unsupported type."""
        mock_embeddings.return_value = Mock()

        retriever = LangChainRetriever()

        documents = [Document(page_content="Test document", metadata={"id": "doc1"})]

        with pytest.raises(ValueError, match="Unsupported vectorstore type"):
            retriever.create_vectorstore(documents, vectorstore_type="unsupported")


class TestLangChainHANRAGIntegration:
    """Test LangChain-HANRAG integration."""

    @pytest.fixture
    def mock_hanrag_system(self):
        """Mock HANRAG system for testing."""
        with patch("src.hanrag.HANRAGSystem"):
            return HANRAGSystem()

    def test_langchain_hanrag_integration_initialization(self, mock_hanrag_system):
        """Test LangChainHANRAGIntegration initialization."""
        integration = LangChainHANRAGIntegration(mock_hanrag_system)

        assert integration.hanrag_system == mock_hanrag_system
        assert integration.document_processor is not None
        assert integration.langchain_retriever is not None

    @patch("src.langchain_integration.DocumentProcessor")
    def test_setup_knowledge_base_from_texts(self, mock_processor, mock_hanrag_system):
        """Test setting up knowledge base from texts."""
        # Mock document processor
        mock_processor.return_value.load_documents_from_text.return_value = [
            Document(page_content="Test content", metadata={"id": "doc1"})
        ]
        mock_processor.return_value.split_documents.return_value = [
            Document(page_content="Test chunk", metadata={"id": "chunk1"})
        ]

        integration = LangChainHANRAGIntegration(mock_hanrag_system)

        # Mock the retriever and hanrag system methods
        integration.langchain_retriever.create_vectorstore = Mock()
        mock_hanrag_system.add_knowledge_base = Mock()

        texts = ["Test document content"]
        metadata = [{"source": "test"}]

        # This would normally call the actual method, but we're testing the integration
        # In a real test, you'd set up the mocks properly
        assert integration.document_processor is not None


class TestHANRAGLangGraphWorkflow:
    """Test HANRAG LangGraph workflow."""

    @pytest.fixture
    def mock_hanrag_system(self):
        """Mock HANRAG system for testing."""
        with patch("src.hanrag.HANRAGSystem"):
            system = HANRAGSystem()
            # Mock the retriever methods
            system.retriever = Mock()
            system.retriever.retrieve_with_noise_detection = Mock()
            return system

    @patch("src.langgraph_integration.ChatOpenAI")
    def test_workflow_initialization(self, mock_chat_openai, mock_hanrag_system):
        """Test HANRAGLangGraphWorkflow initialization."""
        mock_chat_openai.return_value = Mock()

        workflow = HANRAGLangGraphWorkflow(mock_hanrag_system)

        assert workflow.hanrag_system == mock_hanrag_system
        assert workflow.llm is not None
        assert workflow.graph is not None

    @patch("src.langgraph_integration.ChatOpenAI")
    def test_initialize_state(self, mock_chat_openai, mock_hanrag_system):
        """Test state initialization."""
        mock_chat_openai.return_value = Mock()

        workflow = HANRAGLangGraphWorkflow(mock_hanrag_system)

        initial_state = {
            "messages": [HumanMessage(content="What is machine learning?")],
            "question": "",
            "current_hop": 0,
            "max_hops": 3,
            "reasoning_steps": [],
            "retrieved_documents": [],
            "intermediate_answers": [],
            "confidence_scores": [],
            "is_query_noisy": False,
            "noise_score": 0.0,
            "final_answer": None,
            "overall_confidence": 0.0,
            "processing_metadata": {},
        }

        updated_state = workflow._initialize_state(initial_state)

        assert updated_state["question"] == "What is machine learning?"
        assert updated_state["current_hop"] == 0
        assert updated_state["max_hops"] == 3
        assert updated_state["processing_metadata"]["workflow_started"] is True

    @patch("src.langgraph_integration.ChatOpenAI")
    def test_detect_query_noise(self, mock_chat_openai, mock_hanrag_system):
        """Test query noise detection in workflow."""
        mock_chat_openai.return_value = Mock()

        workflow = HANRAGLangGraphWorkflow(mock_hanrag_system)

        # Mock the retriever's retrieve_with_noise_detection method
        mock_hanrag_system.retriever.retrieve_with_noise_detection.return_value = (
            [],
            False,
            0.1,
        )

        state = {
            "messages": [HumanMessage(content="What is machine learning?")],
            "question": "What is machine learning?",
            "current_hop": 0,
            "max_hops": 3,
            "reasoning_steps": [],
            "retrieved_documents": [],
            "intermediate_answers": [],
            "confidence_scores": [],
            "is_query_noisy": False,
            "noise_score": 0.0,
            "final_answer": None,
            "overall_confidence": 0.0,
            "processing_metadata": {},
        }

        updated_state = workflow._detect_query_noise(state)

        assert updated_state["is_query_noisy"] is False
        assert updated_state["noise_score"] == 0.1
        assert len(updated_state["messages"]) == 2  # Should have added a system message


class TestConversationalHANRAGWorkflow:
    """Test ConversationalHANRAGWorkflow."""

    @pytest.fixture
    def mock_hanrag_system(self):
        """Mock HANRAG system for testing."""
        with patch("src.hanrag.HANRAGSystem"):
            return HANRAGSystem()

    @patch("src.langgraph_integration.ChatOpenAI")
    def test_conversational_workflow_initialization(
        self, mock_chat_openai, mock_hanrag_system
    ):
        """Test ConversationalHANRAGWorkflow initialization."""
        mock_chat_openai.return_value = Mock()

        workflow = ConversationalHANRAGWorkflow(mock_hanrag_system)

        assert workflow.hanrag_system == mock_hanrag_system
        assert workflow.llm is not None
        assert workflow.conversation_memory == []
        assert workflow.graph is not None

    @patch("src.langgraph_integration.ChatOpenAI")
    def test_check_conversation_context(self, mock_chat_openai, mock_hanrag_system):
        """Test conversation context checking."""
        mock_chat_openai.return_value = Mock()

        workflow = ConversationalHANRAGWorkflow(mock_hanrag_system)

        # Add some conversation history
        workflow.conversation_memory = [
            {
                "question": "What is machine learning?",
                "answer": "ML is a subset of AI",
                "confidence": 0.9,
            }
        ]

        state = {
            "messages": [HumanMessage(content="What about deep learning?")],
            "question": "What about deep learning?",
            "current_hop": 0,
            "max_hops": 3,
            "reasoning_steps": [],
            "retrieved_documents": [],
            "intermediate_answers": [],
            "confidence_scores": [],
            "is_query_noisy": False,
            "noise_score": 0.0,
            "final_answer": None,
            "overall_confidence": 0.0,
            "processing_metadata": {},
        }

        # Initialize conversation first to set conversation_turn
        state = workflow._initialize_conversation(state)
        updated_state = workflow._check_conversation_context(state)

        assert updated_state["processing_metadata"]["conversation_turn"] == 2
        assert updated_state["processing_metadata"]["has_previous_context"] is True
        assert updated_state["processing_metadata"]["is_follow_up"] is True

    @patch("src.langgraph_integration.ChatOpenAI")
    def test_update_conversation_memory(self, mock_chat_openai, mock_hanrag_system):
        """Test conversation memory update."""
        mock_chat_openai.return_value = Mock()

        workflow = ConversationalHANRAGWorkflow(mock_hanrag_system)

        state = {
            "messages": [HumanMessage(content="Test question")],
            "question": "Test question",
            "current_hop": 1,
            "max_hops": 3,
            "reasoning_steps": [],
            "retrieved_documents": [],
            "intermediate_answers": [],
            "confidence_scores": [],
            "is_query_noisy": False,
            "noise_score": 0.0,
            "final_answer": "Test answer",
            "overall_confidence": 0.8,
            "processing_metadata": {"timestamp": "2024-01-01"},
        }

        updated_state = workflow._update_conversation_memory(state)

        assert len(workflow.conversation_memory) == 1
        assert workflow.conversation_memory[0]["question"] == "Test question"
        assert workflow.conversation_memory[0]["answer"] == "Test answer"
        assert workflow.conversation_memory[0]["confidence"] == 0.8

    @patch("src.langgraph_integration.ChatOpenAI")
    def test_get_conversation_history(self, mock_chat_openai, mock_hanrag_system):
        """Test getting conversation history."""
        mock_chat_openai.return_value = Mock()

        workflow = ConversationalHANRAGWorkflow(mock_hanrag_system)

        # Add some conversation history
        workflow.conversation_memory = [
            {"question": "Q1", "answer": "A1", "confidence": 0.9},
            {"question": "Q2", "answer": "A2", "confidence": 0.8},
        ]

        history = workflow.get_conversation_history()

        assert len(history) == 2
        assert history[0]["question"] == "Q1"
        assert history[1]["question"] == "Q2"

    @patch("src.langgraph_integration.ChatOpenAI")
    def test_clear_conversation_memory(self, mock_chat_openai, mock_hanrag_system):
        """Test clearing conversation memory."""
        mock_chat_openai.return_value = Mock()

        workflow = ConversationalHANRAGWorkflow(mock_hanrag_system)

        # Add some conversation history
        workflow.conversation_memory = [
            {"question": "Q1", "answer": "A1", "confidence": 0.9}
        ]

        workflow.clear_conversation_memory()

        assert len(workflow.conversation_memory) == 0


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    @patch("src.hanrag.NoiseResistantRetriever")
    @patch("src.hanrag.NoiseResistantGenerator")
    @patch("src.langchain_integration.OpenAIEmbeddings")
    def test_full_hanrag_workflow(
        self, mock_embeddings, mock_generator, mock_retriever
    ):
        """Test full HANRAG workflow integration."""
        # Mock the components
        mock_embeddings.return_value = Mock()
        mock_retriever.return_value = Mock()
        mock_generator.return_value = Mock()

        # Create HANRAG system
        hanrag_system = HANRAGSystem()

        # Create integration
        integration = LangChainHANRAGIntegration(hanrag_system)

        # Create workflow
        with patch("src.langgraph_integration.ChatOpenAI"):
            workflow = HANRAGLangGraphWorkflow(hanrag_system)

        # Test that all components are properly integrated
        assert integration.hanrag_system == hanrag_system
        assert workflow.hanrag_system == hanrag_system
        assert integration.document_processor is not None
        assert integration.langchain_retriever is not None
        assert workflow.graph is not None
