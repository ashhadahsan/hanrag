"""
Example script demonstrating how to evaluate and compare traditional RAG with HANRAG.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.rag_evaluator import RAGEvaluator
from src.models import Document


def create_sample_documents():
    """Create sample documents for evaluation."""
    documents = [
        Document(
            page_content="""
            Machine learning is a subset of artificial intelligence (AI) that enables computers to learn 
            and improve from experience without being explicitly programmed. It focuses on the development 
            of computer programs that can access data and use it to learn for themselves.
            """,
            metadata={"id": "ml_intro", "type": "text", "topic": "machine_learning", "source": "textbook"}
        ),
        Document(
            page_content="""
            Deep learning is a subset of machine learning that uses artificial neural networks with 
            multiple layers (deep neural networks) to model and understand complex patterns in data. 
            It has been particularly successful in areas like image recognition, natural language processing, 
            and speech recognition.
            """,
            metadata={"id": "dl_intro", "type": "text", "topic": "deep_learning", "source": "textbook"}
        ),
        Document(
            page_content="""
            Natural Language Processing (NLP) is a field of artificial intelligence that focuses on 
            the interaction between computers and humans through natural language. The ultimate objective 
            of NLP is to read, decipher, understand, and make sense of human language in a valuable way.
            """,
            metadata={"id": "nlp_intro", "type": "text", "topic": "nlp", "source": "textbook"}
        ),
        Document(
            page_content="""
            Computer vision is a field of artificial intelligence that trains computers to interpret 
            and understand the visual world. Using digital images from cameras and videos and deep learning 
            models, machines can accurately identify and classify objects and react to what they see.
            """,
            metadata={"id": "cv_intro", "type": "text", "topic": "computer_vision", "source": "textbook"}
        ),
        Document(
            page_content="""
            Reinforcement learning is an area of machine learning concerned with how software agents 
            should take actions in an environment in order to maximize the notion of cumulative reward. 
            It differs from supervised learning in that labeled input/output pairs need not be presented.
            """,
            metadata={"id": "rl_intro", "type": "text", "topic": "reinforcement_learning", "source": "textbook"}
        ),
        Document(
            page_content="""
            Supervised learning is the machine learning task of learning a function that maps an input 
            to an output based on example input-output pairs. It infers a function from labeled training 
            data consisting of a set of training examples.
            """,
            metadata={"id": "sl_intro", "type": "text", "topic": "supervised_learning", "source": "textbook"}
        ),
        Document(
            page_content="""
            Unsupervised learning is a type of machine learning that looks for previously undetected 
            patterns in a data set with no pre-existing labels and with a minimum of human supervision. 
            It is used to draw inferences from datasets consisting of input data without labeled responses.
            """,
            metadata={"id": "ul_intro", "type": "text", "topic": "unsupervised_learning", "source": "textbook"}
        ),
        Document(
            page_content="""
            Artificial neural networks are computing systems inspired by biological neural networks. 
            These systems learn to perform tasks by considering examples, generally without being programmed 
            with task-specific rules. They can adapt to changing input and generate the best possible result.
            """,
            metadata={"id": "ann_intro", "type": "text", "topic": "neural_networks", "source": "textbook"}
        )
    ]
    return documents


def create_test_questions_and_answers():
    """Create test questions and ground truth answers."""
    questions = [
        "What is machine learning?",
        "How does deep learning relate to machine learning?",
        "What is the difference between supervised and unsupervised learning?",
        "What is natural language processing?",
        "How do neural networks work?",
        "What is computer vision used for?",
        "What is reinforcement learning?",
        "What are the main types of machine learning?",
        "How does deep learning differ from traditional machine learning?",
        "What is the relationship between AI, ML, and deep learning?"
    ]
    
    ground_truths = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
        "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model complex patterns in data.",
        "Supervised learning uses labeled training data to learn input-output mappings, while unsupervised learning finds patterns in data without labeled examples.",
        "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and humans through natural language.",
        "Neural networks are computing systems inspired by biological neural networks that learn to perform tasks by considering examples without being programmed with task-specific rules.",
        "Computer vision is used to train computers to interpret and understand the visual world, identifying and classifying objects from digital images and videos.",
        "Reinforcement learning is concerned with how software agents should take actions in an environment to maximize cumulative reward.",
        "The main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning.",
        "Deep learning uses neural networks with multiple layers to process complex patterns, while traditional ML uses simpler algorithms and feature engineering.",
        "Deep learning is a subset of machine learning, which is a subset of artificial intelligence. AI is the broadest field, ML is a subset of AI, and deep learning is a subset of ML."
    ]
    
    return questions, ground_truths


def run_basic_evaluation():
    """Run a basic evaluation comparing traditional RAG with HANRAG."""
    print("🚀 Starting RAG vs HANRAG Evaluation")
    print("=" * 50)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key before running the evaluation")
        return
    
    # Initialize evaluator
    print("📊 Initializing evaluator...")
    evaluator = RAGEvaluator(
        model_name="gpt-3.5-turbo",
        embedding_model="text-embedding-ada-002"
    )
    
    # Create sample data
    print("📚 Creating sample documents and test questions...")
    documents = create_sample_documents()
    questions, ground_truths = create_test_questions_and_answers()
    
    # Setup knowledge base
    print("🔧 Setting up knowledge base...")
    evaluator.setup_knowledge_base(documents)
    
    # Run evaluation
    print("⚡ Running evaluation...")
    results = evaluator.run_evaluation(
        questions=questions,
        ground_truths=ground_truths,
        save_results=True,
        output_dir="evaluation_results"
    )
    
    # Print summary
    print("\n📈 Evaluation Results Summary:")
    evaluator.print_summary()
    
    print(f"\n📁 Results saved to: evaluation_results/")
    print("   - evaluation_results.json: Detailed results")
    print("   - metrics_summary.csv: Metrics summary")
    print("   - comparison_report.md: Human-readable report")
    print("   - *.png: Visualization charts")
    
    return results


def run_noise_resistance_evaluation():
    """Run evaluation focusing on noise resistance capabilities."""
    print("\n🔇 Running Noise Resistance Evaluation")
    print("=" * 50)
    
    # Create noisy versions of questions
    clean_questions = [
        "What is machine learning?",
        "How does deep learning work?",
        "What is natural language processing?"
    ]
    
    noisy_questions = [
        "What is that thing about computers learning stuff and getting smarter?",
        "How does that deep learning thing work with all those layers?",
        "What is that NLP thing that deals with human language?"
    ]
    
    ground_truths = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        "Deep learning uses neural networks with multiple layers to process complex patterns in data.",
        "Natural Language Processing is a field of AI that focuses on computer-human interaction through natural language."
    ]
    
    evaluator = RAGEvaluator()
    documents = create_sample_documents()
    evaluator.setup_knowledge_base(documents)
    
    # Evaluate clean questions
    print("🧹 Evaluating clean questions...")
    clean_results = evaluator.run_evaluation(
        questions=clean_questions,
        ground_truths=ground_truths,
        save_results=False
    )
    
    # Evaluate noisy questions
    print("🔊 Evaluating noisy questions...")
    noisy_results = evaluator.run_evaluation(
        questions=noisy_questions,
        ground_truths=ground_truths,
        save_results=False
    )
    
    # Compare results
    print("\n📊 Noise Resistance Analysis:")
    print("-" * 30)
    
    for i, (clean_q, noisy_q) in enumerate(zip(clean_questions, noisy_questions)):
        clean_answer = clean_results["hanrag_results"][i]["answer"]
        noisy_answer = noisy_results["hanrag_results"][i]["answer"]
        clean_conf = clean_results["hanrag_results"][i]["confidence"]
        noisy_conf = noisy_results["hanrag_results"][i]["confidence"]
        
        print(f"\nQuestion {i+1}:")
        print(f"  Clean: {clean_q}")
        print(f"  Noisy: {noisy_q}")
        print(f"  Clean Confidence: {clean_conf:.3f}")
        print(f"  Noisy Confidence: {noisy_conf:.3f}")
        print(f"  Confidence Drop: {clean_conf - noisy_conf:.3f}")


def run_multi_hop_evaluation():
    """Run evaluation focusing on multi-hop reasoning capabilities."""
    print("\n🔄 Running Multi-hop Reasoning Evaluation")
    print("=" * 50)
    
    # Multi-hop questions that require reasoning across multiple documents
    multi_hop_questions = [
        "What is the relationship between machine learning and neural networks?",
        "How do supervised learning and deep learning relate to each other?",
        "What are the connections between AI, machine learning, and computer vision?",
        "How does reinforcement learning differ from supervised and unsupervised learning?",
        "What is the hierarchy of AI, machine learning, deep learning, and neural networks?"
    ]
    
    ground_truths = [
        "Neural networks are a key component of machine learning, particularly in deep learning, where they are used to model complex patterns in data.",
        "Deep learning is a subset of machine learning that can use supervised learning techniques with neural networks to learn from labeled data.",
        "Computer vision is a field of AI that often uses machine learning techniques, including deep learning with neural networks, to process visual information.",
        "Reinforcement learning is a separate paradigm from supervised and unsupervised learning, focusing on learning through interaction with an environment to maximize rewards.",
        "The hierarchy is: AI (broadest) > Machine Learning (subset of AI) > Deep Learning (subset of ML) > Neural Networks (tool used in deep learning)."
    ]
    
    evaluator = RAGEvaluator()
    documents = create_sample_documents()
    evaluator.setup_knowledge_base(documents)
    
    print("🧠 Evaluating multi-hop reasoning questions...")
    results = evaluator.run_evaluation(
        questions=multi_hop_questions,
        ground_truths=ground_truths,
        save_results=True,
        output_dir="multi_hop_evaluation_results"
    )
    
    # Analyze reasoning chains
    print("\n🔗 Multi-hop Reasoning Analysis:")
    print("-" * 40)
    
    for i, question in enumerate(multi_hop_questions):
        hanrag_result = results["hanrag_results"][i]
        reasoning_steps = hanrag_result.get("reasoning_chain", [])
        
        print(f"\nQuestion {i+1}: {question}")
        print(f"Number of reasoning steps: {len(reasoning_steps)}")
        print(f"Final confidence: {hanrag_result['confidence']:.3f}")
        
        if reasoning_steps:
            print("Reasoning chain:")
            for j, step in enumerate(reasoning_steps):
                print(f"  Step {j+1}: {step.get('reasoning', 'No reasoning provided')[:100]}...")


def main():
    """Main function to run all evaluations."""
    print("🎯 HANRAG Evaluation Suite")
    print("=" * 60)
    print("This script demonstrates comprehensive evaluation of HANRAG vs Traditional RAG")
    print("=" * 60)
    
    try:
        # Run basic evaluation
        basic_results = run_basic_evaluation()
        
        # Run noise resistance evaluation
        run_noise_resistance_evaluation()
        
        # Run multi-hop evaluation
        run_multi_hop_evaluation()
        
        print("\n✅ All evaluations completed successfully!")
        print("\n📋 Summary of generated files:")
        print("   - evaluation_results/: Basic comparison results")
        print("   - multi_hop_evaluation_results/: Multi-hop reasoning results")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        print("Please check your OpenAI API key and try again.")


if __name__ == "__main__":
    main()
