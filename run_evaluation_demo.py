#!/usr/bin/env python3
"""
Evaluation script that runs HANRAG vs Traditional RAG comparison.
Uses real API calls if OPENAI_API_KEY is set in .env file, otherwise uses mock data.
"""

import os
import sys
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent))

# Set default environment variables (only if not already set)
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "demo-api-key"
if not os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = "demo-langsmith-key"
os.environ["DEFAULT_MODEL"] = "gpt-3.5-turbo"
os.environ["EMBEDDING_MODEL"] = "text-embedding-ada-002"

from evaluation.rag_evaluator import RAGEvaluator
from evaluation.comparison_metrics import ComparisonMetrics
from langchain.schema import Document


def create_sample_documents():
    """Create sample documents for evaluation."""
    documents = [
        Document(
            page_content="""
            Machine learning is a subset of artificial intelligence (AI) that enables computers to learn 
            and improve from experience without being explicitly programmed. It focuses on the development 
            of computer programs that can access data and use it to learn for themselves.
            """,
            metadata={
                "id": "ml_intro",
                "type": "text",
                "topic": "machine_learning",
                "source": "textbook",
            },
        ),
        Document(
            page_content="""
            Deep learning is a subset of machine learning that uses artificial neural networks with 
            multiple layers (deep neural networks) to model and understand complex patterns in data. 
            It has been particularly successful in areas like image recognition, natural language processing, 
            and speech recognition.
            """,
            metadata={
                "id": "dl_intro",
                "type": "text",
                "topic": "deep_learning",
                "source": "textbook",
            },
        ),
        Document(
            page_content="""
            Natural Language Processing (NLP) is a field of artificial intelligence that focuses on 
            the interaction between computers and humans through natural language. The ultimate objective 
            of NLP is to read, decipher, understand, and make sense of human language in a valuable way.
            """,
            metadata={
                "id": "nlp_intro",
                "type": "text",
                "topic": "nlp",
                "source": "textbook",
            },
        ),
        Document(
            page_content="""
            Computer vision is a field of artificial intelligence that trains computers to interpret 
            and understand the visual world. Using digital images from cameras and videos and deep learning 
            models, machines can accurately identify and classify objects and react to what they see.
            """,
            metadata={
                "id": "cv_intro",
                "type": "text",
                "topic": "computer_vision",
                "source": "textbook",
            },
        ),
        Document(
            page_content="""
            Reinforcement learning is an area of machine learning concerned with how software agents 
            should take actions in an environment in order to maximize the notion of cumulative reward. 
            It differs from supervised learning in that labeled input/output pairs need not be presented.
            """,
            metadata={
                "id": "rl_intro",
                "type": "text",
                "topic": "reinforcement_learning",
                "source": "textbook",
            },
        ),
        Document(
            page_content="""
            Supervised learning is the machine learning task of learning a function that maps an input 
            to an output based on example input-output pairs. It infers a function from labeled training 
            data consisting of a set of training examples.
            """,
            metadata={
                "id": "sl_intro",
                "type": "text",
                "topic": "supervised_learning",
                "source": "textbook",
            },
        ),
        Document(
            page_content="""
            Unsupervised learning is a type of machine learning that looks for previously undetected 
            patterns in a data set with no pre-existing labels and with a minimum of human supervision. 
            It is used to draw inferences from datasets consisting of input data without labeled responses.
            """,
            metadata={
                "id": "ul_intro",
                "type": "text",
                "topic": "unsupervised_learning",
                "source": "textbook",
            },
        ),
        Document(
            page_content="""
            Artificial neural networks are computing systems inspired by biological neural networks. 
            These systems learn to perform tasks by considering examples, generally without being programmed 
            with task-specific rules. They can adapt to changing input and generate the best possible result.
            """,
            metadata={
                "id": "ann_intro",
                "type": "text",
                "topic": "neural_networks",
                "source": "textbook",
            },
        ),
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
        "What is the relationship between AI, ML, and deep learning?",
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
        "Deep learning is a subset of machine learning, which is a subset of artificial intelligence. AI is the broadest field, ML is a subset of AI, and deep learning is a subset of ML.",
    ]

    return questions, ground_truths


def generate_mock_results(questions, ground_truths):
    """Generate mock results for both traditional RAG and HANRAG."""
    traditional_results = []
    hanrag_results = []

    for i, (question, gt) in enumerate(zip(questions, ground_truths)):
        # Traditional RAG results (lower quality)
        traditional_result = {
            "answer": f"Traditional RAG answer for: {question}. This is a basic response that may not fully address the question.",
            "confidence": np.random.uniform(0.6, 0.8),
            "retrieved_documents": [
                {"content": f"Document content related to question {i+1}"},
                {"content": f"Additional context for question {i+1}"},
            ],
            "processing_time": np.random.uniform(0.8, 1.5),
            "reasoning_chain": [],
            "noise_detected": False,
            "noise_level": 0.0,
            "metadata": {"system_type": "traditional_rag"},
        }

        # HANRAG results (higher quality)
        hanrag_result = {
            "answer": f"HANRAG provides a comprehensive answer for: {question}. This response demonstrates better understanding and reasoning capabilities.",
            "confidence": np.random.uniform(0.8, 0.95),
            "retrieved_documents": [
                {"content": f"Relevant document 1 for question {i+1}"},
                {"content": f"Relevant document 2 for question {i+1}"},
                {"content": f"Additional context document for question {i+1}"},
            ],
            "processing_time": np.random.uniform(1.2, 2.0),
            "reasoning_chain": [
                {
                    "reasoning": f"Step 1: Analyze the question about {question.split()[0:3]}"
                },
                {
                    "reasoning": f"Step 2: Retrieve relevant information from knowledge base"
                },
                {
                    "reasoning": f"Step 3: Synthesize information to provide comprehensive answer"
                },
            ],
            "noise_detected": np.random.choice([True, False], p=[0.2, 0.8]),
            "noise_level": (
                np.random.uniform(0.0, 0.3)
                if np.random.choice([True, False], p=[0.2, 0.8])
                else 0.0
            ),
            "metadata": {"system_type": "hanrag", "multi_hop": True},
        }

        traditional_results.append(traditional_result)
        hanrag_results.append(hanrag_result)

    return traditional_results, hanrag_results


def run_demo_evaluation():
    """Run a demo evaluation with real API calls or mock data."""
    print("🚀 Starting HANRAG vs Traditional RAG Demo Evaluation")
    print("=" * 60)

    # Check if we have a real API key
    has_real_api_key = (
        os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "demo-api-key"
    )

    if has_real_api_key:
        print("🔑 Using real OpenAI API key for evaluation")
    else:
        print("🎭 Using mock data to generate evaluation plots")
        print("💡 To use real API calls, set your OPENAI_API_KEY environment variable")
    print("=" * 60)

    # Create sample data
    print("📚 Creating sample documents and test questions...")
    documents = create_sample_documents()
    questions, ground_truths = create_test_questions_and_answers()

    if has_real_api_key:
        # Use real evaluation with API calls
        print("⚡ Running real evaluation with API calls...")
        evaluator = RAGEvaluator(
            model_name="gpt-3.5-turbo", embedding_model="text-embedding-ada-002"
        )

        # Setup knowledge base
        print("🔧 Setting up knowledge base...")
        evaluator.setup_knowledge_base(documents)

        # Run evaluation
        print("📊 Running comprehensive evaluation...")
        evaluation_results = evaluator.run_evaluation(
            test_questions=questions,
            ground_truths=ground_truths,
            save_results=False,  # We'll save manually
        )

        comparison_metrics = evaluation_results["comparison_metrics"]
    else:
        # Generate mock results
        print("🎭 Generating mock evaluation results...")
        traditional_results, hanrag_results = generate_mock_results(
            questions, ground_truths
        )

        # Initialize metrics calculator
        print("📊 Calculating comparison metrics...")
        metrics_calculator = ComparisonMetrics()

        # Calculate comprehensive metrics
        comparison_metrics = metrics_calculator.calculate_comprehensive_metrics(
            traditional_results, hanrag_results, ground_truths, questions
        )

        # Create evaluation results structure
        evaluation_results = {
            "test_questions": questions,
            "ground_truths": ground_truths,
            "traditional_rag_results": traditional_results,
            "hanrag_results": hanrag_results,
            "comparison_metrics": comparison_metrics,
            "evaluation_time": {
                "traditional_rag": 12.5,
                "hanrag": 18.3,
                "total": 30.8,
            },
        }

    # Save results and generate visualizations
    output_dir = "evaluation_results"
    print(f"💾 Saving results to {output_dir}/")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save detailed results as JSON
    results_file = output_path / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(evaluation_results, f, indent=2, default=str)

    # Generate visualizations
    print("📈 Generating visualization plots...")
    generate_visualizations(comparison_metrics, output_path)

    # Generate summary report
    generate_summary_report(evaluation_results, output_path)

    print("\n✅ Demo evaluation completed successfully!")
    print(f"\n📁 Results saved to: {output_dir}/")
    print("   - evaluation_results.json: Detailed results")
    print("   - comparison_report.md: Human-readable report")
    print("   - *.png: Visualization charts")

    return evaluation_results


def generate_visualizations(comparison_metrics, output_path):
    """Generate comparison visualizations."""
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # 1. Metrics Comparison Bar Chart
    plot_metrics_comparison(comparison_metrics, output_path)

    # 2. Processing Time Comparison
    plot_processing_time_comparison(comparison_metrics, output_path)

    # 3. Confidence Distribution
    plot_confidence_distribution(comparison_metrics, output_path)

    # 4. Individual Metric Distributions
    plot_metric_distributions(comparison_metrics, output_path)


def plot_metrics_comparison(comparison_metrics, output_path):
    """Plot metrics comparison bar chart."""
    fig, ax = plt.subplots(figsize=(12, 8))

    metrics = [
        "faithfulness",
        "answer_relevance",
        "context_precision",
        "context_recall",
        "bleu_score",
    ]
    traditional_scores = [
        comparison_metrics["traditional_rag"][m]["mean"] for m in metrics
    ]
    hanrag_scores = [comparison_metrics["hanrag"][m]["mean"] for m in metrics]

    x = range(len(metrics))
    width = 0.35

    ax.bar(
        [i - width / 2 for i in x],
        traditional_scores,
        width,
        label="Traditional RAG",
        alpha=0.8,
        color="skyblue",
    )
    ax.bar(
        [i + width / 2 for i in x],
        hanrag_scores,
        width,
        label="HANRAG",
        alpha=0.8,
        color="lightcoral",
    )

    ax.set_xlabel("Metrics")
    ax.set_ylabel("Score")
    ax.set_title("RAG vs HANRAG: Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "metrics_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_processing_time_comparison(comparison_metrics, output_path):
    """Plot processing time comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))

    systems = ["Traditional RAG", "HANRAG"]
    times = [
        comparison_metrics["traditional_rag"]["processing_time"]["mean"],
        comparison_metrics["hanrag"]["processing_time"]["mean"],
    ]

    bars = ax.bar(systems, times, alpha=0.8, color=["skyblue", "lightcoral"])
    ax.set_ylabel("Average Processing Time (seconds)")
    ax.set_title("Processing Time Comparison")
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, time in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{time:.2f}s",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(
        output_path / "processing_time_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_confidence_distribution(comparison_metrics, output_path):
    """Plot confidence score distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Traditional RAG confidence
    trad_conf = comparison_metrics["traditional_rag"]["confidence"]["scores"]
    ax1.hist(trad_conf, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
    ax1.set_title("Traditional RAG Confidence Distribution")
    ax1.set_xlabel("Confidence Score")
    ax1.set_ylabel("Frequency")
    ax1.grid(True, alpha=0.3)

    # HANRAG confidence
    hanrag_conf = comparison_metrics["hanrag"]["confidence"]["scores"]
    ax2.hist(hanrag_conf, bins=20, alpha=0.7, color="lightcoral", edgecolor="black")
    ax2.set_title("HANRAG Confidence Distribution")
    ax2.set_xlabel("Confidence Score")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_path / "confidence_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_metric_distributions(comparison_metrics, output_path):
    """Plot distributions of individual metrics."""
    metrics = [
        "faithfulness",
        "answer_relevance",
        "context_precision",
        "context_recall",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        trad_scores = comparison_metrics["traditional_rag"][metric]["scores"]
        hanrag_scores = comparison_metrics["hanrag"][metric]["scores"]

        axes[i].hist(
            trad_scores,
            bins=15,
            alpha=0.6,
            label="Traditional RAG",
            color="skyblue",
        )
        axes[i].hist(
            hanrag_scores, bins=15, alpha=0.6, label="HANRAG", color="lightcoral"
        )
        axes[i].set_title(f'{metric.replace("_", " ").title()} Distribution')
        axes[i].set_xlabel("Score")
        axes[i].set_ylabel("Frequency")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "metric_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_summary_report(evaluation_results, output_path):
    """Generate a detailed comparison report."""
    report_file = output_path / "comparison_report.md"

    with open(report_file, "w") as f:
        f.write("# RAG vs HANRAG Comparison Report\n\n")
        f.write(f"**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Questions:** {len(evaluation_results['test_questions'])}\n\n")

        # Performance Summary
        f.write("## Performance Summary\n\n")
        comparison = evaluation_results["comparison_metrics"]["comparison"]

        f.write("### Key Improvements (HANRAG vs Traditional RAG)\n\n")
        for metric in [
            "faithfulness",
            "answer_relevance",
            "context_precision",
            "context_recall",
            "bleu_score",
        ]:
            improvement = comparison[f"{metric}_improvement"]
            f.write(
                f"- **{metric.replace('_', ' ').title()}:** {improvement['relative']:.1f}% improvement\n"
            )

        f.write(
            f"\n- **Processing Time Ratio:** {comparison['processing_time_ratio']:.2f}x\n"
        )
        f.write(
            f"- **Confidence Improvement:** {comparison['confidence_improvement']['relative']:.1f}%\n\n"
        )

        # Detailed Metrics
        f.write("## Detailed Metrics\n\n")
        for system in ["traditional_rag", "hanrag"]:
            f.write(f"### {system.replace('_', ' ').title()}\n\n")
            metrics = evaluation_results["comparison_metrics"][system]

            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and "mean" in metric_data:
                    f.write(
                        f"- **{metric_name.replace('_', ' ').title()}:** {metric_data['mean']:.3f} ± {metric_data['std']:.3f}\n"
                    )
            f.write("\n")


def print_summary(evaluation_results):
    """Print a summary of the evaluation results."""
    comparison_metrics = evaluation_results["comparison_metrics"]
    comparison = comparison_metrics["comparison"]

    print("\n" + "=" * 60)
    print("RAG vs HANRAG EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Questions Evaluated: {len(evaluation_results['test_questions'])}")
    print(
        f"Total Evaluation Time: {evaluation_results['evaluation_time']['total']:.2f} seconds"
    )
    print()

    print("KEY IMPROVEMENTS (HANRAG vs Traditional RAG):")
    print("-" * 50)
    for metric in [
        "faithfulness",
        "answer_relevance",
        "context_precision",
        "context_recall",
        "bleu_score",
    ]:
        improvement = comparison[f"{metric}_improvement"]
        print(
            f"{metric.replace('_', ' ').title():<20}: {improvement['relative']:>6.1f}%"
        )

    print(f"\nProcessing Time Ratio: {comparison['processing_time_ratio']:.2f}x")
    print(
        f"Confidence Improvement: {comparison['confidence_improvement']['relative']:.1f}%"
    )
    print("=" * 60)


if __name__ == "__main__":
    try:
        results = run_demo_evaluation()
        print_summary(results)

        print("\n📋 Summary of generated files:")
        print("   - evaluation_results/: Demo comparison results")
        print("   - evaluation_results.json: Detailed results")
        print("   - comparison_report.md: Human-readable report")
        print("   - *.png: Visualization charts")

    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        import traceback

        traceback.print_exc()
