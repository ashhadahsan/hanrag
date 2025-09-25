"""
Basic usage examples for HANRAG system.
"""

import os
from langchain_core.documents import Document
from src.hanrag import HANRAGSystem
from src.langchain_integration import LangChainHANRAGIntegration
from src.langgraph_integration import HANRAGLangGraphWorkflow


def example_basic_hanrag():
    """Basic HANRAG usage example."""
    print("=== Basic HANRAG Usage ===")

    # Initialize HANRAG system
    hanrag = HANRAGSystem()

    # Create sample documents
    documents = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
            metadata={"id": "doc1", "type": "text", "domain": "technology"},
        ),
        Document(
            page_content="Deep learning is a subset of machine learning that uses neural networks with multiple layers to process complex patterns in data.",
            metadata={"id": "doc2", "type": "text", "domain": "technology"},
        ),
        Document(
            page_content="Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language.",
            metadata={"id": "doc3", "type": "text", "domain": "technology"},
        ),
    ]

    # Add documents to knowledge base
    hanrag.add_knowledge_base(documents)

    # Ask a question
    question = "What is the relationship between machine learning and deep learning?"
    response = hanrag.answer_question(question)

    print(f"Question: {question}")
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence:.3f}")
    print(f"Processing Time: {response.processing_time:.2f}s")
    print(f"Reasoning Steps: {len(response.reasoning_chain)}")
    print(f"Noise Detected: {response.query.noise_detected}")
    print()


def example_multi_hop_reasoning():
    """Multi-hop reasoning example."""
    print("=== Multi-hop Reasoning ===")

    hanrag = HANRAGSystem()

    # More complex documents for multi-hop reasoning
    documents = [
        Document(
            page_content="Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals.",
            metadata={"id": "doc1", "type": "text", "domain": "technology"},
        ),
        Document(
            page_content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            metadata={"id": "doc2", "type": "text", "domain": "technology"},
        ),
        Document(
            page_content="Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
            metadata={"id": "doc3", "type": "text", "domain": "technology"},
        ),
        Document(
            page_content="Neural networks are computing systems inspired by biological neural networks that constitute animal brains.",
            metadata={"id": "doc4", "type": "text", "domain": "technology"},
        ),
        Document(
            page_content="Computer vision is a field of artificial intelligence that trains computers to interpret and understand visual information from the world.",
            metadata={"id": "doc5", "type": "text", "domain": "technology"},
        ),
    ]

    hanrag.add_knowledge_base(documents)

    # Complex multi-hop question
    question = "How do neural networks relate to computer vision in the context of artificial intelligence?"
    response = hanrag.answer_question(question)

    print(f"Question: {question}")
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence:.3f}")

    print("\nReasoning Chain:")
    for i, step in enumerate(response.reasoning_chain, 1):
        print(f"  Step {i}: {step.question}")
        print(f"    Reasoning: {step.reasoning}")
        print(f"    Answer: {step.intermediate_answer}")
        print(f"    Confidence: {step.confidence:.3f}")
        print()
    print()


def example_noise_resistance():
    """Noise resistance example."""
    print("=== Noise Resistance ===")

    hanrag = HANRAGSystem()

    documents = [
        Document(
            page_content="Machine learning algorithms can be categorized into supervised learning, unsupervised learning, and reinforcement learning.",
            metadata={"id": "doc1", "type": "text", "domain": "technology"},
        ),
        Document(
            page_content="Supervised learning uses labeled training data to learn a mapping from inputs to outputs.",
            metadata={"id": "doc2", "type": "text", "domain": "technology"},
        ),
        Document(
            page_content="Unsupervised learning finds hidden patterns in data without labeled examples.",
            metadata={"id": "doc3", "type": "text", "domain": "technology"},
        ),
    ]

    hanrag.add_knowledge_base(documents)

    # Noisy questions
    noisy_questions = [
        "What is that thing about computers learning stuff?",
        "Tell me about machine learning and how it works and stuff",
        "What about supervised learning and unsupervised learning and things like that?",
    ]

    for question in noisy_questions:
        response = hanrag.answer_question(question)
        print(f"Question: {question}")
        print(f"Answer: {response.answer}")
        print(f"Noise Detected: {response.query.noise_detected}")
        print(f"Noise Level: {response.query.noise_level:.3f}")
        print(f"Confidence: {response.confidence:.3f}")
        print()


def example_batch_processing():
    """Batch processing example."""
    print("=== Batch Processing ===")

    hanrag = HANRAGSystem()

    documents = [
        Document(
            page_content="Python is a high-level programming language known for its simplicity and readability.",
            metadata={"id": "doc1", "type": "text", "domain": "programming"},
        ),
        Document(
            page_content="Machine learning libraries in Python include scikit-learn, TensorFlow, and PyTorch.",
            metadata={"id": "doc2", "type": "text", "domain": "programming"},
        ),
        Document(
            page_content="Data science involves extracting insights from data using statistical and computational methods.",
            metadata={"id": "doc3", "type": "text", "domain": "data_science"},
        ),
    ]

    hanrag.add_knowledge_base(documents)

    questions = [
        "What is Python?",
        "What machine learning libraries are available in Python?",
        "What is data science?",
        "How does Python relate to machine learning and data science?",
    ]

    responses = hanrag.batch_answer_questions(questions)

    for question, response in zip(questions, responses):
        print(f"Q: {question}")
        print(f"A: {response.answer}")
        print(f"Confidence: {response.confidence:.3f}")
        print("-" * 50)


def example_langchain_integration():
    """LangChain integration example."""
    print("=== LangChain Integration ===")

    hanrag = HANRAGSystem()
    integration = LangChainHANRAGIntegration(hanrag)

    # Create documents from text
    texts = [
        "LangChain is a framework for developing applications powered by language models.",
        "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        "Vector stores are used to store and retrieve document embeddings for similarity search.",
    ]

    metadata = [
        {"source": "langchain_docs", "type": "framework"},
        {"source": "langgraph_docs", "type": "framework"},
        {"source": "vectorstore_docs", "type": "concept"},
    ]

    documents = integration.document_processor.load_documents_from_text(texts, metadata)
    integration.setup_knowledge_base_from_texts(texts, metadata)

    question = "What is the relationship between LangChain and LangGraph?"
    response = hanrag.answer_question(question)

    print(f"Question: {question}")
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence:.3f}")
    print()


def example_langgraph_workflow():
    """LangGraph workflow example."""
    print("=== LangGraph Workflow ===")

    hanrag = HANRAGSystem()

    documents = [
        Document(
            page_content="Research in artificial intelligence has led to significant advances in machine learning, natural language processing, and computer vision.",
            metadata={"id": "doc1", "type": "research", "domain": "ai"},
        ),
        Document(
            page_content="Machine learning research focuses on developing algorithms that can learn from data and make predictions or decisions.",
            metadata={"id": "doc2", "type": "research", "domain": "ml"},
        ),
        Document(
            page_content="Natural language processing research aims to enable computers to understand, interpret, and generate human language.",
            metadata={"id": "doc3", "type": "research", "domain": "nlp"},
        ),
    ]

    hanrag.add_knowledge_base(documents)

    # Create workflow
    workflow = HANRAGLangGraphWorkflow(hanrag)

    question = "How do machine learning and natural language processing contribute to artificial intelligence research?"
    response = workflow.run_workflow(question)

    print(f"Question: {question}")
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence:.3f}")
    print(f"Processing Time: {response.processing_time:.2f}s")
    print()


def example_system_evaluation():
    """System evaluation example."""
    print("=== System Evaluation ===")

    hanrag = HANRAGSystem()

    documents = [
        Document(
            page_content="The transformer architecture revolutionized natural language processing by introducing attention mechanisms.",
            metadata={"id": "doc1", "type": "technical", "domain": "nlp"},
        ),
        Document(
            page_content="Attention mechanisms allow models to focus on relevant parts of the input sequence when making predictions.",
            metadata={"id": "doc2", "type": "technical", "domain": "nlp"},
        ),
        Document(
            page_content="BERT and GPT are popular transformer-based models that have achieved state-of-the-art results in various NLP tasks.",
            metadata={"id": "doc3", "type": "technical", "domain": "nlp"},
        ),
    ]

    hanrag.add_knowledge_base(documents)

    # Test questions and expected answers
    test_questions = [
        "What is the transformer architecture?",
        "How do attention mechanisms work?",
        "What are some popular transformer models?",
    ]

    expected_answers = [
        "The transformer architecture revolutionized NLP with attention mechanisms",
        "Attention mechanisms allow models to focus on relevant input parts",
        "BERT and GPT are popular transformer-based models",
    ]

    evaluation = hanrag.evaluate_system(test_questions, expected_answers)

    print("Evaluation Results:")
    print(f"Total Questions: {evaluation['total_questions']}")
    print(f"Accuracy: {evaluation['accuracy']:.3f}")
    print(f"Average Confidence: {evaluation['avg_confidence']:.3f}")
    print(f"High Confidence Rate: {evaluation['high_confidence_rate']:.3f}")
    print(f"Noise Detection Rate: {evaluation['noise_detection_rate']:.3f}")
    print(f"Average Processing Time: {evaluation['avg_processing_time']:.2f}s")
    print(f"Average Reasoning Steps: {evaluation['avg_reasoning_steps']:.1f}")
    print()


def main():
    from dotenv import load_dotenv

    load_dotenv()
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY environment variable")
        exit(1)

    # Run examples
    example_basic_hanrag()
    example_multi_hop_reasoning()
    example_noise_resistance()
    example_batch_processing()
    example_langchain_integration()
    example_langgraph_workflow()
    example_system_evaluation()

    print("All examples completed successfully!")


if __name__ == "__main__":
    main()
