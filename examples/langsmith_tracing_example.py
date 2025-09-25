#!/usr/bin/env python3
"""
Example demonstrating LangSmith tracing with HANRAG system.
This example shows how to use the HANRAG system with LangSmith tracing enabled.
"""

import os
import sys
from langchain_core.documents import Document

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hanrag import HANRAGSystem
from src.langchain_integration import LangChainHANRAGIntegration
from src.langgraph_integration import HANRAGLangGraphWorkflow


def main():
    """Main example function."""

    print("🔍 HANRAG System with LangSmith Tracing Example")
    print("=" * 60)

    # Check if LangSmith is configured
    from src.config import config

    if not config.langsmith_tracing or not config.langsmith_api_key:
        print("❌ LangSmith tracing is not configured!")
        print(
            "Please set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in your .env file"
        )
        return

    print("✅ LangSmith tracing is enabled!")
    print(f"Project: {os.getenv('LANGCHAIN_PROJECT', 'Default')}")
    print()

    # Initialize HANRAG system
    print("🚀 Initializing HANRAG System...")
    hanrag = HANRAGSystem()

    # Create sample documents
    documents = [
        Document(
            page_content="Artificial Intelligence (AI) is a broad field of computer science focused on creating systems that can perform tasks that typically require human intelligence.",
            metadata={"id": "ai_intro", "type": "definition", "source": "textbook"},
        ),
        Document(
            page_content="Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed for every task.",
            metadata={"id": "ml_def", "type": "definition", "source": "textbook"},
        ),
        Document(
            page_content="Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data.",
            metadata={"id": "dl_def", "type": "definition", "source": "textbook"},
        ),
        Document(
            page_content="Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and humans through natural language.",
            metadata={"id": "nlp_def", "type": "definition", "source": "textbook"},
        ),
        Document(
            page_content="Computer Vision is a field of AI that trains computers to interpret and understand visual information from the world, such as images and videos.",
            metadata={"id": "cv_def", "type": "definition", "source": "textbook"},
        ),
    ]

    # Add documents to knowledge base
    print("📚 Adding documents to knowledge base...")
    hanrag.add_knowledge_base(documents)
    print(f"✅ Added {len(documents)} documents to knowledge base")
    print()

    # Test questions of different complexity levels
    test_questions = [
        "What is artificial intelligence?",
        "How are machine learning and deep learning related?",
        "What are the main applications of natural language processing?",
        "Compare and contrast computer vision with natural language processing in terms of their goals and applications.",
    ]

    print("❓ Testing different types of questions...")
    print()

    for i, question in enumerate(test_questions, 1):
        print(f"Question {i}: {question}")
        print("-" * 50)

        try:
            # This will automatically use LangSmith tracing
            response = hanrag.answer_question(question)

            print(f"Answer: {response.answer}")
            print(f"Confidence: {response.confidence:.3f}")
            print(f"Query Type: {response.metadata.get('query_type', 'Unknown')}")
            print(f"Processing Time: {response.processing_time:.2f}s")
            print(f"Reasoning Steps: {len(response.reasoning_chain)}")

            # Show token usage if available
            if "total_tokens" in response.metadata:
                print(
                    f"Tokens Used: {response.metadata['total_tokens']} "
                    f"(Prompt: {response.metadata['prompt_tokens']}, "
                    f"Completion: {response.metadata['completion_tokens']})"
                )
                print(f"Cost: ${response.metadata['total_cost']:.4f}")

            print()

        except Exception as e:
            print(f"❌ Error processing question: {str(e)}")
            print()

    print("🎉 Example completed!")
    print("Check your LangSmith dashboard to see the traces for this session.")
    print("Look for traces under the 'HANRAG-System' project.")


def test_langchain_integration():
    """Test LangChain integration with tracing."""
    print("\n🔗 Testing LangChain Integration with Tracing")
    print("=" * 50)

    from src.config import config

    if not config.langsmith_tracing:
        print("LangSmith tracing not enabled, skipping LangChain integration test")
        return

    # Initialize integration
    hanrag = HANRAGSystem()
    integration = LangChainHANRAGIntegration(hanrag)

    # Setup knowledge base using LangChain
    print("📚 Setting up knowledge base using LangChain...")
    test_texts = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "JavaScript is a programming language primarily used for web development and creating interactive web pages.",
        "Machine learning algorithms can be implemented in both Python and JavaScript, though Python is more commonly used.",
    ]

    # Create documents from texts
    documents = integration.document_processor.load_documents_from_text(
        test_texts,
        [
            {"source": f"text_{i}", "type": "programming"}
            for i in range(len(test_texts))
        ],
    )

    # Add to HANRAG system
    integration.hanrag_system.add_knowledge_base(documents)

    # Create vector store for LangChain retriever
    integration.langchain_retriever.create_vectorstore(documents)

    # Test retrieval with tracing
    print("🔍 Testing similarity search with tracing...")
    results = integration.langchain_retriever.similarity_search("What is Python?", k=2)
    print(f"Found {len(results)} relevant documents")

    print("✅ LangChain integration test completed!")


def test_langgraph_workflow():
    """Test LangGraph workflow with tracing."""
    print("\n🔄 Testing LangGraph Workflow with Tracing")
    print("=" * 50)

    from src.config import config

    if not config.langsmith_tracing:
        print("LangSmith tracing not enabled, skipping LangGraph workflow test")
        return

    # Initialize workflow
    hanrag = HANRAGSystem()
    workflow = HANRAGLangGraphWorkflow(hanrag)

    # Add documents
    documents = [
        Document(
            page_content="Climate change refers to long-term shifts in global temperatures and weather patterns.",
            metadata={"id": "climate_1", "type": "science"},
        ),
        Document(
            page_content="Greenhouse gases like carbon dioxide trap heat in the Earth's atmosphere, contributing to global warming.",
            metadata={"id": "climate_2", "type": "science"},
        ),
    ]

    hanrag.add_knowledge_base(documents)

    # Test workflow with tracing
    print("🔄 Running LangGraph workflow...")
    question = "How does climate change relate to greenhouse gases?"

    try:
        response = workflow.run_workflow(question)
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence:.3f}")
        print("✅ LangGraph workflow test completed!")

    except Exception as e:
        print(f"❌ Error in LangGraph workflow: {str(e)}")


if __name__ == "__main__":
    main()
    test_langchain_integration()
    test_langgraph_workflow()
