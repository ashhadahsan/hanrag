"""
Example demonstrating the Revelator components in HANRAG.
"""

import os
from langchain_core.documents import Document
from src.hanrag import HANRAGSystem
from src.models import QueryType


def example_query_routing():
    """Demonstrate query routing with different query types."""
    print("=== Query Routing Examples ===")

    hanrag = HANRAGSystem()

    # Test different query types
    test_queries = [
        ("What is 2+2?", "Straightforward - basic math"),
        ("When was Pan Jianwei born?", "Single-step - factual lookup"),
        (
            "When was Liu Xiang born and when did he retire?",
            "Compound - multiple independent facts",
        ),
        (
            "Who succeeded the first President of Namibia?",
            "Complex - multi-step reasoning",
        ),
    ]

    for query, description in test_queries:
        query_type = hanrag.revelator.route_query(query)
        print(f"Query: {query}")
        print(f"Description: {description}")
        print(f"Classified as: {query_type.value}")
        print("-" * 50)


def example_query_decomposition():
    """Demonstrate query decomposition for compound queries."""
    print("=== Query Decomposition Examples ===")

    hanrag = HANRAGSystem()

    compound_queries = [
        "When was Liu Xiang born and when did he retire?",
        "What are the main features and price of iPhone 15?",
        "What is the capital of France and what is its population?",
    ]

    for query in compound_queries:
        sub_queries = hanrag.revelator.decompose_query(query)
        print(f"Original Query: {query}")
        print("Decomposed into:")
        for i, sub_query in enumerate(sub_queries, 1):
            print(f"  {i}. {sub_query}")
        print("-" * 50)


def example_relevance_discrimination():
    """Demonstrate document relevance discrimination."""
    print("=== Relevance Discrimination Examples ===")

    hanrag = HANRAGSystem()

    query = "When was Liu Xiang born?"
    documents = [
        "Liu Xiang was born on July 13, 1983, in Shanghai, China.",
        "The weather today is sunny and warm with temperatures reaching 25°C.",
        "Liu Xiang is a Chinese hurdler who won Olympic gold in 2004.",
        "Machine learning is a subset of artificial intelligence.",
    ]

    print(f"Query: {query}")
    print("Document Relevance Assessment:")

    for i, doc in enumerate(documents, 1):
        is_relevant = hanrag.revelator.discriminate_relevance(query, doc)
        status = "RELEVANT" if is_relevant else "NOT RELEVANT"
        print(f"  Doc {i}: {status}")
        print(f"    Content: {doc}")
        print()


def example_seed_question_refinement():
    """Demonstrate seed question refinement for complex queries."""
    print("=== Seed Question Refinement Examples ===")

    hanrag = HANRAGSystem()

    complex_query = "Who succeeded the first President of Namibia?"

    print(f"Complex Query: {complex_query}")
    print("Refinement Process:")

    # First step - no previous context
    seed_question = hanrag.revelator.refine_seed_question(complex_query)
    print(f"Step 1 - Initial seed question: {seed_question}")

    # Simulate some reasoning steps
    from src.models import ReasoningStep

    reasoning_steps = [
        ReasoningStep(
            step_number=1,
            question="Who is the first President of Namibia?",
            retrieved_documents=[],
            reasoning="Need to find the first president to determine who succeeded them",
            intermediate_answer="Sam Nujoma",
            confidence=0.9,
        )
    ]

    # Second step - with previous context
    next_seed = hanrag.revelator.refine_seed_question(complex_query, reasoning_steps)
    print(f"Step 2 - Next seed question: {next_seed}")

    # Add more reasoning steps
    reasoning_steps.append(
        ReasoningStep(
            step_number=2,
            question="Who succeeded Sam Nujoma as President of Namibia?",
            retrieved_documents=[],
            reasoning="Now that we know Sam Nujoma was the first president, find his successor",
            intermediate_answer="Hifikepunye Pohamba",
            confidence=0.9,
        )
    )

    # Check if sufficient
    is_sufficient = hanrag.revelator.should_end_reasoning(
        complex_query, reasoning_steps
    )
    print(f"Step 3 - Sufficient information gathered: {is_sufficient}")


def example_full_hanrag_workflow():
    """Demonstrate the complete HANRAG workflow with Revelator."""
    print("=== Complete HANRAG Workflow ===")

    hanrag = HANRAGSystem()

    # Create sample documents
    documents = [
        Document(
            page_content="Liu Xiang was born on July 13, 1983, in Shanghai, China. He is a Chinese hurdler who won the Olympic gold medal in the 110-meter hurdles at the 2004 Athens Olympics.",
            metadata={"id": "doc1", "type": "text", "domain": "sports"},
        ),
        Document(
            page_content="Liu Xiang announced his retirement from professional athletics on April 7, 2015, through a statement posted on his Sina Weibo account.",
            metadata={"id": "doc2", "type": "text", "domain": "sports"},
        ),
        Document(
            page_content="Sam Nujoma was the first President of Namibia, serving from 1990 to 2005. He was succeeded by Hifikepunye Pohamba.",
            metadata={"id": "doc3", "type": "text", "domain": "politics"},
        ),
        Document(
            page_content="Hifikepunye Pohamba served as the second President of Namibia from 2005 to 2015, succeeding Sam Nujoma.",
            metadata={"id": "doc4", "type": "text", "domain": "politics"},
        ),
    ]

    # Add documents to knowledge base
    hanrag.add_knowledge_base(documents)

    # Test different query types
    test_queries = [
        "What is 2+2?",  # Straightforward
        "When was Liu Xiang born?",  # Single-step
        "When was Liu Xiang born and when did he retire?",  # Compound
        "Who succeeded the first President of Namibia?",  # Complex
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("=" * 60)

        response = hanrag.answer_question(query)

        print(f"Query Type: {response.metadata.get('query_type', 'unknown')}")
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence:.3f}")
        print(f"Processing Time: {response.processing_time:.2f}s")
        print(f"Reasoning Steps: {len(response.reasoning_chain)}")

        if response.reasoning_chain:
            print("Reasoning Chain:")
            for i, step in enumerate(response.reasoning_chain, 1):
                print(f"  Step {i}: {step.question}")
                print(f"    Answer: {step.intermediate_answer}")
                print(f"    Confidence: {step.confidence:.3f}")

        print("-" * 60)


def example_parallel_processing():
    """Demonstrate parallel processing for compound queries."""
    print("=== Parallel Processing for Compound Queries ===")

    hanrag = HANRAGSystem()

    # Create documents
    documents = [
        Document(
            page_content="Liu Xiang was born on July 13, 1983, in Shanghai, China.",
            metadata={"id": "doc1", "type": "text", "domain": "sports"},
        ),
        Document(
            page_content="Liu Xiang retired from professional athletics on April 7, 2015.",
            metadata={"id": "doc2", "type": "text", "domain": "sports"},
        ),
        Document(
            page_content="Liu Xiang won Olympic gold in the 110-meter hurdles at the 2004 Athens Olympics.",
            metadata={"id": "doc3", "type": "text", "domain": "sports"},
        ),
    ]

    hanrag.add_knowledge_base(documents)

    # Test compound query
    compound_query = "When was Liu Xiang born and when did he retire?"

    print(f"Compound Query: {compound_query}")

    # Decompose the query
    sub_queries = hanrag.revelator.decompose_query(compound_query)
    print(f"Decomposed into {len(sub_queries)} sub-queries:")
    for i, sub_query in enumerate(sub_queries, 1):
        print(f"  {i}. {sub_query}")

    # Process the compound query
    response = hanrag.answer_question(compound_query)

    print(f"\nFinal Answer: {response.answer}")
    print(f"Processing Type: {response.metadata.get('processing_type', 'unknown')}")
    print(f"Sub-results: {len(response.metadata.get('sub_results', []))}")


if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY environment variable")
        exit(1)

    # Run examples
    example_query_routing()
    print("\n")
    example_query_decomposition()
    print("\n")
    example_relevance_discrimination()
    print("\n")
    example_seed_question_refinement()
    print("\n")
    example_full_hanrag_workflow()
    print("\n")
    example_parallel_processing()

    print("\nAll Revelator examples completed successfully!")
