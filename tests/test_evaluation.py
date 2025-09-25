"""
Tests for the RAG evaluation framework.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from evaluation.rag_evaluator import RAGEvaluator
from evaluation.traditional_rag import TraditionalRAGSystem
from evaluation.comparison_metrics import ComparisonMetrics
from langchain_core.documents import Document


class TestTraditionalRAGSystem:
    """Test traditional RAG system implementation."""

    def test_initialization(self):
        """Test traditional RAG system initialization."""
        rag = TraditionalRAGSystem()
        assert rag.embeddings is not None
        assert rag.llm is not None
        assert rag.text_splitter is not None
        assert rag.vectorstore is None
        assert rag.qa_chain is None

    @patch("evaluation.traditional_rag.OpenAIEmbeddings")
    def test_add_knowledge_base(self, mock_embeddings_class):
        """Test adding documents to knowledge base."""
        # Mock the embeddings instance
        mock_embeddings_instance = Mock()
        mock_embeddings_instance.embed_documents.return_value = [
            [0.1, 0.2, 0.3] * 1536,  # Mock embedding for doc 1
            [0.4, 0.5, 0.6] * 1536,  # Mock embedding for doc 2
        ]
        mock_embeddings_class.return_value = mock_embeddings_instance

        rag = TraditionalRAGSystem()
        documents = [
            Document(
                page_content="Test document 1", metadata={"id": "doc1", "type": "text"}
            ),
            Document(
                page_content="Test document 2", metadata={"id": "doc2", "type": "text"}
            ),
        ]

        rag.add_knowledge_base(documents)
        assert rag.vectorstore is not None
        assert rag.qa_chain is not None

    @patch("evaluation.traditional_rag.TraditionalRAGSystem.__init__")
    def test_answer_question(self, mock_init):
        """Test answering a question."""
        # Mock the initialization to avoid complex dependencies
        mock_init.return_value = None

        rag = TraditionalRAGSystem()

        # Mock all the components
        rag.embeddings = Mock()
        rag.llm = Mock()
        rag.text_splitter = Mock()
        rag.vectorstore = Mock()
        rag.qa_chain = Mock()

        # Mock the QA chain response
        rag.qa_chain.invoke.return_value = {"result": "Test answer"}

        # Mock the vectorstore response
        rag.vectorstore.similarity_search.return_value = [
            Mock(page_content="Test content", metadata={"id": "doc1"})
        ]

        result = rag.answer_question("What is the test content?")

        assert result["answer"] == "Test answer"
        assert result["confidence"] == 0.8
        assert result["processing_time"] > 0
        assert len(result["retrieved_documents"]) > 0

    @patch("evaluation.traditional_rag.TraditionalRAGSystem.__init__")
    def test_batch_answer_questions(self, mock_init):
        """Test batch question answering."""
        # Mock the initialization to avoid complex dependencies
        mock_init.return_value = None

        rag = TraditionalRAGSystem()

        # Mock the answer_question method
        rag.answer_question = Mock(
            return_value={
                "answer": "Test answer",
                "confidence": 0.8,
                "processing_time": 1.0,
                "retrieved_documents": [],
                "reasoning_chain": [],
                "noise_detected": False,
                "noise_level": 0.0,
                "metadata": {},
            }
        )

        questions = ["Question 1", "Question 2"]
        results = rag.batch_answer_questions(questions)

        assert len(results) == 2
        assert all(result["answer"] == "Test answer" for result in results)


class TestComparisonMetrics:
    """Test comparison metrics calculation."""

    def test_initialization(self):
        """Test metrics calculator initialization."""
        metrics = ComparisonMetrics()
        assert metrics.embedding_model is not None
        assert metrics.smoothing is not None

    def test_calculate_faithfulness(self):
        """Test faithfulness calculation."""
        metrics = ComparisonMetrics()

        answer = "Machine learning is a subset of artificial intelligence"
        context = "Artificial intelligence includes machine learning and deep learning"

        faithfulness = metrics.calculate_faithfulness(answer, context)
        assert 0 <= faithfulness <= 1

    def test_calculate_answer_relevance(self):
        """Test answer relevance calculation."""
        metrics = ComparisonMetrics()

        question = "What is machine learning?"
        answer = "Machine learning is a subset of AI that enables computers to learn"

        relevance = metrics.calculate_answer_relevance(question, answer)
        assert 0 <= relevance <= 1

    def test_calculate_context_precision(self):
        """Test context precision calculation."""
        metrics = ComparisonMetrics()

        question = "What is machine learning?"
        docs = [
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks",
            "Natural language processing is another AI field",
        ]

        precision = metrics.calculate_context_precision(question, docs)
        assert 0 <= precision <= 1

    def test_calculate_context_recall(self):
        """Test context recall calculation."""
        metrics = ComparisonMetrics()

        ground_truth = "Machine learning is a subset of artificial intelligence"
        docs = [
            "Machine learning is a subset of artificial intelligence that enables computers to learn from data"
        ]

        recall = metrics.calculate_context_recall(ground_truth, docs)
        assert 0 <= recall <= 1

    def test_calculate_bleu_score(self):
        """Test BLEU score calculation."""
        metrics = ComparisonMetrics()

        reference = "Machine learning is a subset of artificial intelligence"
        candidate = "Machine learning is part of artificial intelligence"

        bleu = metrics.calculate_bleu_score(reference, candidate)
        assert 0 <= bleu <= 1

    def test_calculate_rouge_score(self):
        """Test ROUGE score calculation."""
        metrics = ComparisonMetrics()

        reference = "Machine learning is a subset of artificial intelligence"
        candidate = "Machine learning is part of artificial intelligence"

        rouge = metrics.calculate_rouge_score(reference, candidate)
        assert "rouge_1" in rouge
        assert "rouge_2" in rouge
        assert "rouge_l" in rouge
        assert all(0 <= score <= 1 for score in rouge.values())

    def test_calculate_noise_resistance(self):
        """Test noise resistance calculation."""
        metrics = ComparisonMetrics()

        noisy_question = "What is that thing about computers learning stuff?"
        clean_question = "What is machine learning?"
        noisy_answer = "Machine learning is when computers learn from data"
        clean_answer = "Machine learning is a subset of AI that enables computers to learn from data"

        resistance = metrics.calculate_noise_resistance(
            noisy_question, clean_question, noisy_answer, clean_answer
        )
        assert 0 <= resistance <= 1

    def test_calculate_comprehensive_metrics(self):
        """Test comprehensive metrics calculation."""
        metrics = ComparisonMetrics()

        # Mock results
        traditional_results = [
            {
                "answer": "Machine learning is a subset of AI",
                "confidence": 0.8,
                "processing_time": 1.0,
                "retrieved_documents": [
                    {
                        "content": "Machine learning is a subset of artificial intelligence"
                    }
                ],
            }
        ]

        hanrag_results = [
            {
                "answer": "Machine learning is a subset of artificial intelligence",
                "confidence": 0.9,
                "processing_time": 2.0,
                "retrieved_documents": [
                    {
                        "content": "Machine learning is a subset of artificial intelligence"
                    }
                ],
            }
        ]

        ground_truths = ["Machine learning is a subset of artificial intelligence"]
        questions = ["What is machine learning?"]

        comprehensive_metrics = metrics.calculate_comprehensive_metrics(
            traditional_results, hanrag_results, ground_truths, questions
        )

        assert "traditional_rag" in comprehensive_metrics
        assert "hanrag" in comprehensive_metrics
        assert "comparison" in comprehensive_metrics

        # Check that all required metrics are present
        for system in ["traditional_rag", "hanrag"]:
            system_metrics = comprehensive_metrics[system]
            assert "faithfulness" in system_metrics
            assert "answer_relevance" in system_metrics
            assert "context_precision" in system_metrics
            assert "context_recall" in system_metrics
            assert "bleu_score" in system_metrics
            assert "processing_time" in system_metrics
            assert "confidence" in system_metrics


class TestRAGEvaluator:
    """Test RAG evaluator."""

    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = RAGEvaluator()
        assert evaluator.traditional_rag is not None
        assert evaluator.hanrag is not None
        assert evaluator.metrics is not None
        assert evaluator.evaluation_results == {}
        assert evaluator.comparison_metrics == {}

    def test_setup_knowledge_base(self):
        """Test knowledge base setup."""
        evaluator = RAGEvaluator()
        documents = [
            Document(
                page_content="Test document 1", metadata={"id": "doc1", "type": "text"}
            ),
            Document(
                page_content="Test document 2", metadata={"id": "doc2", "type": "text"}
            ),
        ]

        # Mock the add_knowledge_base methods
        evaluator.traditional_rag.add_knowledge_base = Mock()
        evaluator.hanrag.add_knowledge_base = Mock()

        evaluator.setup_knowledge_base(documents)

        evaluator.traditional_rag.add_knowledge_base.assert_called_once_with(documents)
        evaluator.hanrag.add_knowledge_base.assert_called_once_with(documents)

    @patch("evaluation.rag_evaluator.RAGEvaluator._save_results")
    def test_run_evaluation(self, mock_save_results):
        """Test running evaluation."""
        evaluator = RAGEvaluator()

        # Mock the systems
        evaluator.traditional_rag.batch_answer_questions = Mock(
            return_value=[
                {
                    "answer": "Traditional answer",
                    "confidence": 0.8,
                    "processing_time": 1.0,
                    "retrieved_documents": [{"content": "test content"}],
                    "reasoning_chain": [],
                    "noise_detected": False,
                    "noise_level": 0.0,
                    "metadata": {},
                }
            ]
        )

        evaluator.hanrag.batch_answer_questions = Mock(
            return_value=[
                {
                    "answer": "HANRAG answer",
                    "confidence": 0.9,
                    "processing_time": 2.0,
                    "retrieved_documents": [{"content": "test content"}],
                    "reasoning_chain": [],
                    "noise_detected": False,
                    "noise_level": 0.0,
                    "metadata": {},
                }
            ]
        )

        # Mock the metrics calculation
        evaluator.metrics.calculate_comprehensive_metrics = Mock(
            return_value={
                "traditional_rag": {"faithfulness": {"mean": 0.8, "std": 0.1}},
                "hanrag": {"faithfulness": {"mean": 0.9, "std": 0.1}},
                "comparison": {
                    "faithfulness_improvement": {"absolute": 0.1, "relative": 12.5}
                },
            }
        )

        test_questions = ["What is machine learning?"]
        ground_truths = ["Machine learning is a subset of AI"]

        results = evaluator.run_evaluation(
            test_questions, ground_truths, save_results=False
        )

        assert "test_questions" in results
        assert "ground_truths" in results
        assert "traditional_rag_results" in results
        assert "hanrag_results" in results
        assert "comparison_metrics" in results
        assert "evaluation_time" in results

    def test_get_summary_statistics(self):
        """Test getting summary statistics."""
        evaluator = RAGEvaluator()

        # Mock evaluation results
        evaluator.evaluation_results = {
            "test_questions": ["Question 1", "Question 2"],
            "evaluation_time": {"traditional_rag": 1.0, "hanrag": 2.0, "total": 3.0},
        }

        evaluator.comparison_metrics = {
            "comparison": {
                "faithfulness_improvement": {"absolute": 0.1, "relative": 12.5},
                "answer_relevance_improvement": {"absolute": 0.05, "relative": 6.25},
                "context_precision_improvement": {"absolute": 0.08, "relative": 10.0},
                "context_recall_improvement": {"absolute": 0.12, "relative": 15.0},
                "bleu_score_improvement": {"absolute": 0.06, "relative": 7.5},
                "processing_time_ratio": 2.0,
                "confidence_improvement": {"absolute": 0.1, "relative": 12.5},
            }
        }

        summary = evaluator.get_summary_statistics()

        assert summary["total_questions"] == 2
        assert summary["evaluation_time"]["total"] == 3.0
        assert "key_improvements" in summary
        assert summary["processing_time_ratio"] == 2.0

    def test_save_results(self):
        """Test saving results to files."""
        evaluator = RAGEvaluator()

        # Mock evaluation results
        evaluator.evaluation_results = {
            "test_questions": ["Question 1"],
            "ground_truths": ["Answer 1"],
            "traditional_rag_results": [
                {"answer": "Traditional answer", "confidence": 0.8}
            ],
            "hanrag_results": [{"answer": "HANRAG answer", "confidence": 0.9}],
            "comparison_metrics": {
                "traditional_rag": {
                    "faithfulness": {
                        "mean": 0.8,
                        "std": 0.1,
                        "scores": [0.7, 0.8, 0.9],
                    },
                    "answer_relevance": {
                        "mean": 0.75,
                        "std": 0.1,
                        "scores": [0.7, 0.75, 0.8],
                    },
                    "context_precision": {
                        "mean": 0.7,
                        "std": 0.1,
                        "scores": [0.6, 0.7, 0.8],
                    },
                    "context_recall": {
                        "mean": 0.65,
                        "std": 0.1,
                        "scores": [0.6, 0.65, 0.7],
                    },
                    "bleu_score": {"mean": 0.6, "std": 0.1, "scores": [0.5, 0.6, 0.7]},
                    "processing_time": {
                        "mean": 1.0,
                        "std": 0.1,
                        "scores": [0.9, 1.0, 1.1],
                    },
                    "confidence": {"mean": 0.8, "std": 0.1, "scores": [0.7, 0.8, 0.9]},
                },
                "hanrag": {
                    "faithfulness": {
                        "mean": 0.9,
                        "std": 0.1,
                        "scores": [0.8, 0.9, 1.0],
                    },
                    "answer_relevance": {
                        "mean": 0.85,
                        "std": 0.1,
                        "scores": [0.8, 0.85, 0.9],
                    },
                    "context_precision": {
                        "mean": 0.8,
                        "std": 0.1,
                        "scores": [0.7, 0.8, 0.9],
                    },
                    "context_recall": {
                        "mean": 0.75,
                        "std": 0.1,
                        "scores": [0.7, 0.75, 0.8],
                    },
                    "bleu_score": {"mean": 0.7, "std": 0.1, "scores": [0.6, 0.7, 0.8]},
                    "processing_time": {
                        "mean": 2.0,
                        "std": 0.1,
                        "scores": [1.9, 2.0, 2.1],
                    },
                    "confidence": {"mean": 0.9, "std": 0.1, "scores": [0.8, 0.9, 1.0]},
                },
                "comparison": {
                    "faithfulness_improvement": {"absolute": 0.1, "relative": 12.5},
                    "answer_relevance_improvement": {
                        "absolute": 0.05,
                        "relative": 6.25,
                    },
                    "context_precision_improvement": {
                        "absolute": 0.08,
                        "relative": 10.0,
                    },
                    "context_recall_improvement": {"absolute": 0.12, "relative": 15.0},
                    "bleu_score_improvement": {"absolute": 0.06, "relative": 7.5},
                    "processing_time_ratio": 2.0,
                    "confidence_improvement": {"absolute": 0.1, "relative": 12.5},
                },
            },
        }

        # Also set comparison_metrics directly for the _save_metrics_summary method
        evaluator.comparison_metrics = evaluator.evaluation_results[
            "comparison_metrics"
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator._save_results(temp_dir)

            output_path = Path(temp_dir)
            assert (output_path / "evaluation_results.json").exists()
            assert (output_path / "metrics_summary.csv").exists()
            assert (output_path / "comparison_report.md").exists()

    def test_print_summary(self, capsys):
        """Test printing summary."""
        evaluator = RAGEvaluator()

        # Mock evaluation results
        evaluator.evaluation_results = {
            "test_questions": ["Question 1"],
            "evaluation_time": {"total": 3.0},
        }

        evaluator.comparison_metrics = {
            "comparison": {
                "faithfulness_improvement": {"absolute": 0.1, "relative": 12.5},
                "answer_relevance_improvement": {"absolute": 0.05, "relative": 6.25},
                "context_precision_improvement": {"absolute": 0.08, "relative": 10.0},
                "context_recall_improvement": {"absolute": 0.12, "relative": 15.0},
                "bleu_score_improvement": {"absolute": 0.06, "relative": 7.5},
                "processing_time_ratio": 2.0,
                "confidence_improvement": {"absolute": 0.1, "relative": 12.5},
            }
        }

        evaluator.print_summary()
        captured = capsys.readouterr()

        assert "RAG vs HANRAG EVALUATION SUMMARY" in captured.out
        assert "Total Questions Evaluated: 1" in captured.out
        assert "Processing Time Ratio: 2.00x" in captured.out


@pytest.mark.integration
class TestEvaluationIntegration:
    """Integration tests for the evaluation framework."""

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            Document(
                page_content="Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
                metadata={"id": "doc1", "type": "text", "topic": "machine_learning"},
            ),
            Document(
                page_content="Deep learning is a subset of machine learning that uses neural networks with multiple layers to process complex patterns in data.",
                metadata={"id": "doc2", "type": "text", "topic": "deep_learning"},
            ),
            Document(
                page_content="Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language.",
                metadata={"id": "doc3", "type": "text", "topic": "nlp"},
            ),
            Document(
                page_content="Computer vision is a field of artificial intelligence that trains computers to interpret and understand visual information from the world.",
                metadata={"id": "doc4", "type": "text", "topic": "computer_vision"},
            ),
        ]

    @pytest.fixture
    def sample_questions_and_answers(self):
        """Sample questions and ground truth answers."""
        return {
            "questions": [
                "What is machine learning?",
                "How does deep learning relate to machine learning?",
                "What is the difference between NLP and computer vision?",
                "What are the main fields of artificial intelligence?",
            ],
            "ground_truths": [
                "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
                "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
                "NLP focuses on human language interaction while computer vision focuses on visual information interpretation.",
                "The main fields of AI include machine learning, deep learning, NLP, and computer vision.",
            ],
        }

    @pytest.mark.skip(
        reason="Complex integration test with extensive mocking requirements"
    )
    @patch("evaluation.traditional_rag.TraditionalRAGSystem.__init__")
    @patch("src.hanrag.HANRAGSystem.__init__")
    def test_full_evaluation_workflow(
        self,
        mock_hanrag_init,
        mock_trad_init,
        sample_documents,
        sample_questions_and_answers,
    ):
        """Test the complete evaluation workflow."""
        # Mock the initialization to avoid complex dependencies
        mock_trad_init.return_value = None
        mock_hanrag_init.return_value = None

        evaluator = RAGEvaluator()

        # Mock the necessary attributes for TraditionalRAGSystem
        evaluator.traditional_rag.text_splitter = Mock()
        evaluator.traditional_rag.text_splitter.split_documents.return_value = (
            sample_documents
        )

        # Mock embeddings with proper return values
        mock_embeddings = Mock()
        mock_embeddings.embed_documents.return_value = [
            [0.1, 0.2, 0.3] * 1536 for _ in range(len(sample_documents))
        ]
        evaluator.traditional_rag.embeddings = mock_embeddings
        evaluator.traditional_rag.llm = Mock()
        evaluator.traditional_rag.prompt_template = Mock()
        evaluator.traditional_rag.vectorstore = Mock()
        evaluator.traditional_rag.qa_chain = Mock()

        # Mock the necessary attributes for HANRAGSystem
        evaluator.hanrag.retriever = Mock()
        evaluator.hanrag.generator = Mock()

        # Setup knowledge base
        evaluator.setup_knowledge_base(sample_documents)

        # Mock the batch answer methods
        evaluator.traditional_rag.batch_answer_questions = Mock(
            return_value=[
                {
                    "answer": "Machine learning is a subset of AI that enables computers to learn from data.",
                    "confidence": 0.8,
                    "processing_time": 1.0,
                    "retrieved_documents": [
                        {"content": sample_documents[0].page_content}
                    ],
                    "reasoning_chain": [],
                    "noise_detected": False,
                    "noise_level": 0.0,
                    "metadata": {},
                }
            ]
            * 4
        )

        evaluator.hanrag.batch_answer_questions = Mock(
            return_value=[
                {
                    "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
                    "confidence": 0.9,
                    "processing_time": 2.0,
                    "retrieved_documents": [
                        {"content": sample_documents[0].page_content}
                    ],
                    "reasoning_chain": [
                        {"reasoning": "Step 1: Identify the main concept"}
                    ],
                    "noise_detected": False,
                    "noise_level": 0.0,
                    "metadata": {},
                }
            ]
            * 4
        )

        # Run evaluation
        results = evaluator.run_evaluation(
            sample_questions_and_answers["questions"],
            sample_questions_and_answers["ground_truths"],
            save_results=False,
        )

        # Verify results structure
        assert "test_questions" in results
        assert "ground_truths" in results
        assert "traditional_rag_results" in results
        assert "hanrag_results" in results
        assert "comparison_metrics" in results
        assert "evaluation_time" in results

        # Verify that both systems were called
        evaluator.traditional_rag.batch_answer_questions.assert_called_once()
        evaluator.hanrag.batch_answer_questions.assert_called_once()

        # Verify metrics were calculated
        assert "traditional_rag" in results["comparison_metrics"]
        assert "hanrag" in results["comparison_metrics"]
        assert "comparison" in results["comparison_metrics"]
