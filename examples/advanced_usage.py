"""
Advanced usage examples for HANRAG system.
"""

import os
import sys
import time
from typing import List, Dict, Any

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from src.hanrag import HANRAGSystem
from src.langchain_integration import LangChainHANRAGIntegration
from src.langgraph_integration import ConversationalHANRAGWorkflow
from src.models import HANRAGResponse, RetrievalResult


def example_custom_heuristic_rules():
    """Example of using custom heuristic rules."""
    print("=== Custom Heuristic Rules ===")

    hanrag = HANRAGSystem()

    # Add custom documents with specific metadata
    documents = [
        Document(
            page_content="Recent research shows that transformer models have achieved breakthrough results in natural language processing tasks.",
            metadata={
                "id": "doc1",
                "type": "scientific_paper",
                "domain": "nlp",
                "year": "2024",
                "recent": True,
                "citations": 150,
            },
        ),
        Document(
            page_content="Older research from 2020 showed that RNNs were commonly used for sequence modeling tasks.",
            metadata={
                "id": "doc2",
                "type": "scientific_paper",
                "domain": "nlp",
                "year": "2020",
                "recent": False,
                "citations": 50,
            },
        ),
        Document(
            page_content="Machine learning applications in healthcare have shown promising results for medical diagnosis.",
            metadata={
                "id": "doc3",
                "type": "news_article",
                "domain": "healthcare",
                "year": "2023",
                "recent": True,
                "citations": 75,
            },
        ),
    ]

    hanrag.add_knowledge_base(documents)

    # Questions that should benefit from different heuristic rules
    questions = [
        "What are the latest advances in natural language processing?",  # Should prefer recent docs
        "How has NLP research evolved over time?",  # Should consider both recent and older docs
        "What are the applications of machine learning in healthcare?",  # Should prefer healthcare domain
    ]

    for question in questions:
        response = hanrag.answer_question(question)
        print(f"Question: {question}")
        print(f"Answer: {response.answer}")
        print(f"Retrieved Documents: {len(response.retrieved_documents)}")
        print(f"Confidence: {response.confidence:.3f}")
        print()


def example_confidence_threshold_tuning():
    """Example of tuning confidence thresholds."""
    print("=== Confidence Threshold Tuning ===")

    hanrag = HANRAGSystem()

    documents = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"id": "doc1", "type": "text"},
        ),
        Document(
            page_content="Deep learning uses neural networks with multiple layers.",
            metadata={"id": "doc2", "type": "text"},
        ),
        Document(
            page_content="Natural language processing enables computers to understand human language.",
            metadata={"id": "doc3", "type": "text"},
        ),
    ]

    hanrag.add_knowledge_base(documents)

    question = "What is the relationship between machine learning, deep learning, and natural language processing?"

    # Test different confidence thresholds
    confidence_thresholds = [0.5, 0.7, 0.9]

    for threshold in confidence_thresholds:
        # Temporarily modify the confidence threshold
        from src.config import config

        original_threshold = config.confidence_threshold
        config.confidence_threshold = threshold

        response = hanrag.answer_question(question)

        print(f"Confidence Threshold: {threshold}")
        print(f"Reasoning Steps: {len(response.reasoning_chain)}")
        print(f"Final Confidence: {response.confidence:.3f}")
        print(f"Processing Time: {response.processing_time:.2f}s")
        print()

        # Restore original threshold
        config.confidence_threshold = original_threshold


def example_noise_detection_analysis():
    """Example of analyzing noise detection capabilities."""
    print("=== Noise Detection Analysis ===")

    hanrag = HANRAGSystem()

    documents = [
        Document(
            page_content="Artificial intelligence is a broad field that encompasses machine learning, natural language processing, and computer vision.",
            metadata={"id": "doc1", "type": "text"},
        ),
        Document(
            page_content="Machine learning algorithms can be supervised, unsupervised, or reinforcement learning approaches.",
            metadata={"id": "doc2", "type": "text"},
        ),
    ]

    hanrag.add_knowledge_base(documents)

    # Test different types of queries
    test_queries = [
        # Clear queries
        "What is artificial intelligence?",
        "What are the types of machine learning?",
        # Slightly noisy queries
        "What is AI and how does it work?",
        "Tell me about machine learning types",
        # Very noisy queries
        "What is that thing about computers being smart?",
        "Tell me about AI and ML and stuff like that",
        "How does that machine learning thing work and what are the different types?",
    ]

    for query in test_queries:
        response = hanrag.answer_question(query)
        print(f"Query: {query}")
        print(f"Noise Detected: {response.query.noise_detected}")
        print(f"Noise Level: {response.query.noise_level:.3f}")
        print(f"Answer Quality: {response.confidence:.3f}")
        print(f"Answer: {response.answer[:100]}...")
        print()


def example_performance_benchmarking():
    """Example of performance benchmarking."""
    print("=== Performance Benchmarking ===")

    hanrag = HANRAGSystem()

    # Create a larger document set
    documents = []
    for i in range(50):
        doc = Document(
            page_content=f"Document {i}: This is a comprehensive document about artificial intelligence, machine learning, and related technologies. "
            * 20,
            metadata={"id": f"doc_{i}", "type": "text", "domain": "ai"},
        )
        documents.append(doc)

    hanrag.add_knowledge_base(documents)

    # Benchmark questions
    benchmark_questions = [
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

    # Measure performance
    start_time = time.time()
    responses = hanrag.batch_answer_questions(benchmark_questions)
    total_time = time.time() - start_time

    # Calculate statistics
    total_questions = len(benchmark_questions)
    avg_time_per_question = total_time / total_questions
    avg_confidence = sum(r.confidence for r in responses) / total_questions
    avg_reasoning_steps = (
        sum(len(r.reasoning_chain) for r in responses) / total_questions
    )
    high_confidence_count = sum(1 for r in responses if r.confidence > 0.8)

    print(f"Performance Results:")
    print(f"Total Questions: {total_questions}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Time per Question: {avg_time_per_question:.2f}s")
    print(f"Average Confidence: {avg_confidence:.3f}")
    print(f"Average Reasoning Steps: {avg_reasoning_steps:.1f}")
    print(
        f"High Confidence Answers: {high_confidence_count}/{total_questions} ({high_confidence_count/total_questions*100:.1f}%)"
    )
    print()


def example_conversational_workflow():
    """Example of conversational workflow."""
    print("=== Conversational Workflow ===")

    hanrag = HANRAGSystem()

    documents = [
        Document(
            page_content="Python is a versatile programming language used in data science, web development, and artificial intelligence.",
            metadata={"id": "doc1", "type": "text"},
        ),
        Document(
            page_content="NumPy is a fundamental library for numerical computing in Python, providing support for large arrays and matrices.",
            metadata={"id": "doc2", "type": "text"},
        ),
        Document(
            page_content="Pandas is a powerful data manipulation library built on top of NumPy, providing data structures for structured data analysis.",
            metadata={"id": "doc3", "type": "text"},
        ),
        Document(
            page_content="Scikit-learn is a machine learning library for Python that provides simple and efficient tools for data mining and analysis.",
            metadata={"id": "doc4", "type": "text"},
        ),
    ]

    hanrag.add_knowledge_base(documents)

    # Create conversational workflow
    conversation = ConversationalHANRAGWorkflow(hanrag)

    # Simulate a conversation
    conversation_turns = [
        "What is Python?",
        "What libraries are commonly used with Python?",
        "How do NumPy and Pandas relate to each other?",
        "What about machine learning libraries?",
        "How does scikit-learn fit into the Python ecosystem?",
    ]

    for i, question in enumerate(conversation_turns, 1):
        print(f"Turn {i}: {question}")
        response = conversation.run_conversation(question)
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence:.3f}")
        print()

    # Show conversation history
    history = conversation.get_conversation_history()
    print(f"Conversation History ({len(history)} turns):")
    for i, turn in enumerate(history, 1):
        print(f"  {i}. Q: {turn['question']}")
        print(f"     A: {turn['answer'][:100]}...")
        print(f"     Confidence: {turn['confidence']:.3f}")
    print()


def example_custom_retrieval_strategies():
    """Example of custom retrieval strategies."""
    print("=== Custom Retrieval Strategies ===")

    hanrag = HANRAGSystem()

    # Documents with different characteristics
    documents = [
        Document(
            page_content="Short document about AI.",
            metadata={"id": "doc1", "type": "text", "length": "short"},
        ),
        Document(
            page_content="This is a medium-length document that provides a comprehensive overview of machine learning concepts, including supervised learning, unsupervised learning, and reinforcement learning approaches. It covers the basic principles and applications of each type.",
            metadata={"id": "doc2", "type": "text", "length": "medium"},
        ),
        Document(
            page_content="This is a very long and detailed document that provides an extensive analysis of artificial intelligence, machine learning, deep learning, natural language processing, computer vision, and other related fields. It includes historical context, current developments, future prospects, technical details, implementation considerations, and real-world applications. The document covers various algorithms, architectures, frameworks, and tools used in the field. It also discusses challenges, limitations, and ethical considerations in AI development and deployment. This comprehensive resource serves as a complete guide for understanding the breadth and depth of artificial intelligence research and applications."
            * 3,
            metadata={"id": "doc3", "type": "text", "length": "long"},
        ),
    ]

    hanrag.add_knowledge_base(documents)

    # Test different retrieval strategies
    questions = [
        "What is machine learning?",
        "Give me a detailed explanation of AI concepts",
        "What are the different types of learning in ML?",
    ]

    for question in questions:
        # Test with different retrieval parameters
        print(f"Question: {question}")

        # Standard retrieval
        response = hanrag.answer_question(question)
        print(f"Standard Retrieval:")
        print(f"  Documents: {len(response.retrieved_documents)}")
        print(f"  Confidence: {response.confidence:.3f}")
        print(f"  Answer: {response.answer[:100]}...")
        print()


def example_error_handling():
    """Example of error handling and edge cases."""
    print("=== Error Handling ===")

    hanrag = HANRAGSystem()

    # Test with empty knowledge base
    print("Testing with empty knowledge base:")
    try:
        response = hanrag.answer_question("What is machine learning?")
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence:.3f}")
    except Exception as e:
        print(f"Error: {e}")
    print()

    # Add minimal documents
    documents = [
        Document(
            page_content="Machine learning is a subset of AI.", metadata={"id": "doc1"}
        )
    ]
    hanrag.add_knowledge_base(documents)

    # Test edge cases
    edge_case_questions = [
        "",  # Empty question
        "?",  # Just punctuation
        "a" * 1000,  # Very long question
        "What is machine learning? " * 100,  # Repetitive question
    ]

    for question in edge_case_questions:
        print(f"Testing edge case: '{question[:50]}...'")
        try:
            response = hanrag.answer_question(question)
            print(f"  Answer: {response.answer[:100]}...")
            print(f"  Confidence: {response.confidence:.3f}")
        except Exception as e:
            print(f"  Error: {e}")
        print()


def example_custom_models():
    """Example of using custom models and configurations."""
    print("=== Custom Models and Configuration ===")

    # Create HANRAG system with custom configuration
    hanrag = HANRAGSystem(
        model_name="gpt-4",  # Use GPT-4 instead of default
        embedding_model="text-embedding-3-large",  # Use larger embedding model
    )

    documents = [
        Document(
            page_content="Advanced machine learning techniques include ensemble methods, deep learning, and reinforcement learning.",
            metadata={"id": "doc1", "type": "text"},
        ),
        Document(
            page_content="Ensemble methods combine multiple models to improve prediction accuracy and robustness.",
            metadata={"id": "doc2", "type": "text"},
        ),
    ]

    hanrag.add_knowledge_base(documents)

    # Test with custom parameters
    question = "What are advanced machine learning techniques and how do ensemble methods work?"

    # Custom retrieval parameters
    original_top_k = hanrag.retriever.top_k_documents
    hanrag.retriever.top_k_documents = 10  # Retrieve more documents

    response = hanrag.answer_question(question)

    print(f"Question: {question}")
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence:.3f}")
    print(f"Retrieved Documents: {len(response.retrieved_documents)}")
    print(f"Reasoning Steps: {len(response.reasoning_chain)}")

    # Restore original parameters
    hanrag.retriever.top_k_documents = original_top_k
    print()


def main():
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY environment variable")
        exit(1)

    # Run advanced examples
    example_custom_heuristic_rules()
    example_confidence_threshold_tuning()
    example_noise_detection_analysis()
    example_performance_benchmarking()
    example_conversational_workflow()
    example_custom_retrieval_strategies()
    example_error_handling()
    example_custom_models()

    print("All advanced examples completed successfully!")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    main()
