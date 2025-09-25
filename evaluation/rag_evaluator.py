"""
Main RAG evaluator for comparing traditional RAG with HANRAG.
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from evaluation.traditional_rag import TraditionalRAGSystem
from .comparison_metrics import ComparisonMetrics
from src.hanrag import HANRAGSystem
from langchain.schema import Document


class RAGEvaluator:
    """
    Comprehensive evaluator for comparing traditional RAG with HANRAG.
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-ada-002",
        metrics_model: str = "all-MiniLM-L6-v2",
    ):
        """Initialize the RAG evaluator."""
        self.traditional_rag = TraditionalRAGSystem(model_name, embedding_model)
        self.hanrag = HANRAGSystem(model_name, embedding_model)
        self.metrics = ComparisonMetrics(metrics_model)

        # Results storage
        self.evaluation_results = {}
        self.comparison_metrics = {}

    def setup_knowledge_base(self, documents: List[Document]):
        """Setup knowledge base for both systems."""
        print("Setting up knowledge base for both systems...")

        # Add documents to traditional RAG
        self.traditional_rag.add_knowledge_base(documents)

        # Add documents to HANRAG
        self.hanrag.add_knowledge_base(documents)

        print(f"Knowledge base setup complete. Added {len(documents)} documents.")

    def run_evaluation(
        self,
        test_questions: List[str],
        ground_truths: List[str],
        save_results: bool = True,
        output_dir: str = "evaluation_results",
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation comparing both systems.

        Args:
            test_questions: List of test questions
            ground_truths: List of ground truth answers
            save_results: Whether to save results to files
            output_dir: Directory to save results

        Returns:
            Dictionary with evaluation results
        """
        if len(test_questions) != len(ground_truths):
            raise ValueError("Number of questions and ground truths must match")

        print(f"Running evaluation on {len(test_questions)} questions...")

        # Run traditional RAG
        print("Evaluating Traditional RAG...")
        traditional_start = time.time()
        traditional_results = self.traditional_rag.batch_answer_questions(
            test_questions
        )
        traditional_time = time.time() - traditional_start

        # Run HANRAG
        print("Evaluating HANRAG...")
        hanrag_start = time.time()
        hanrag_results = self.hanrag.batch_answer_questions(test_questions)
        hanrag_time = time.time() - hanrag_start

        # Calculate comprehensive metrics
        print("Calculating comparison metrics...")
        self.comparison_metrics = self.metrics.calculate_comprehensive_metrics(
            traditional_results, hanrag_results, ground_truths, test_questions
        )

        # Store results
        self.evaluation_results = {
            "test_questions": test_questions,
            "ground_truths": ground_truths,
            "traditional_rag_results": traditional_results,
            "hanrag_results": hanrag_results,
            "comparison_metrics": self.comparison_metrics,
            "evaluation_time": {
                "traditional_rag": traditional_time,
                "hanrag": hanrag_time,
                "total": traditional_time + hanrag_time,
            },
        }

        # Save results if requested
        if save_results:
            self._save_results(output_dir)

        print("Evaluation complete!")
        return self.evaluation_results

    def _save_results(self, output_dir: str):
        """Save evaluation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save detailed results as JSON
        results_file = output_path / "evaluation_results.json"
        with open(results_file, "w") as f:
            # Convert results to serializable format
            serializable_results = self._make_serializable(self.evaluation_results)
            json.dump(serializable_results, f, indent=2)

        # Save metrics summary as CSV
        self._save_metrics_summary(output_path)

        # Generate comparison report
        self._generate_comparison_report(output_path)

        # Generate visualizations
        self._generate_visualizations(output_path)

        print(f"Results saved to {output_path}")

    def _make_serializable(self, obj):
        """Convert objects to JSON serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, "__dict__"):
            return self._make_serializable(obj.__dict__)
        else:
            return obj

    def _save_metrics_summary(self, output_path: Path):
        """Save metrics summary as CSV."""
        summary_data = []

        for system in ["traditional_rag", "hanrag"]:
            metrics = self.comparison_metrics[system]
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and "mean" in metric_data:
                    summary_data.append(
                        {
                            "system": system,
                            "metric": metric_name,
                            "mean": metric_data["mean"],
                            "std": metric_data["std"],
                        }
                    )

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path / "metrics_summary.csv", index=False)

    def _generate_comparison_report(self, output_path: Path):
        """Generate a detailed comparison report."""
        report_file = output_path / "comparison_report.md"

        with open(report_file, "w") as f:
            f.write("# RAG vs HANRAG Comparison Report\n\n")
            f.write(f"**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                f"**Total Questions:** {len(self.evaluation_results['test_questions'])}\n\n"
            )

            # Performance Summary
            f.write("## Performance Summary\n\n")
            comparison = self.comparison_metrics["comparison"]

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
                metrics = self.comparison_metrics[system]

                for metric_name, metric_data in metrics.items():
                    if isinstance(metric_data, dict) and "mean" in metric_data:
                        f.write(
                            f"- **{metric_name.replace('_', ' ').title()}:** {metric_data['mean']:.3f} ± {metric_data['std']:.3f}\n"
                        )
                f.write("\n")

            # Sample Results
            f.write("## Sample Results\n\n")
            for i, (question, gt, trad_result, hanrag_result) in enumerate(
                zip(
                    self.evaluation_results["test_questions"][:3],
                    self.evaluation_results["ground_truths"][:3],
                    self.evaluation_results["traditional_rag_results"][:3],
                    self.evaluation_results["hanrag_results"][:3],
                )
            ):
                f.write(f"### Question {i+1}\n\n")
                f.write(f"**Question:** {question}\n\n")
                f.write(f"**Ground Truth:** {gt}\n\n")
                f.write(f"**Traditional RAG Answer:** {trad_result['answer']}\n\n")
                f.write(f"**HANRAG Answer:** {hanrag_result['answer']}\n\n")
                f.write(
                    f"**Traditional RAG Confidence:** {trad_result['confidence']:.3f}\n\n"
                )
                f.write(f"**HANRAG Confidence:** {hanrag_result['confidence']:.3f}\n\n")
                f.write("---\n\n")

    def _generate_visualizations(self, output_path: Path):
        """Generate comparison visualizations."""
        plt.style.use("seaborn-v0_8")

        # Set up the plotting style
        sns.set_palette("husl")

        # 1. Metrics Comparison Bar Chart
        self._plot_metrics_comparison(output_path)

        # 2. Processing Time Comparison
        self._plot_processing_time_comparison(output_path)

        # 3. Confidence Distribution
        self._plot_confidence_distribution(output_path)

        # 4. Individual Metric Distributions
        self._plot_metric_distributions(output_path)

    def _plot_metrics_comparison(self, output_path: Path):
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
            self.comparison_metrics["traditional_rag"][m]["mean"] for m in metrics
        ]
        hanrag_scores = [self.comparison_metrics["hanrag"][m]["mean"] for m in metrics]

        x = range(len(metrics))
        width = 0.35

        ax.bar(
            [i - width / 2 for i in x],
            traditional_scores,
            width,
            label="Traditional RAG",
            alpha=0.8,
        )
        ax.bar(
            [i + width / 2 for i in x], hanrag_scores, width, label="HANRAG", alpha=0.8
        )

        ax.set_xlabel("Metrics")
        ax.set_ylabel("Score")
        ax.set_title("RAG vs HANRAG: Metrics Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_path / "metrics_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_processing_time_comparison(self, output_path: Path):
        """Plot processing time comparison."""
        fig, ax = plt.subplots(figsize=(8, 6))

        systems = ["Traditional RAG", "HANRAG"]
        times = [
            self.comparison_metrics["traditional_rag"]["processing_time"]["mean"],
            self.comparison_metrics["hanrag"]["processing_time"]["mean"],
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

    def _plot_confidence_distribution(self, output_path: Path):
        """Plot confidence score distributions."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Traditional RAG confidence
        trad_conf = self.comparison_metrics["traditional_rag"]["confidence"]["scores"]
        ax1.hist(trad_conf, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        ax1.set_title("Traditional RAG Confidence Distribution")
        ax1.set_xlabel("Confidence Score")
        ax1.set_ylabel("Frequency")
        ax1.grid(True, alpha=0.3)

        # HANRAG confidence
        hanrag_conf = self.comparison_metrics["hanrag"]["confidence"]["scores"]
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

    def _plot_metric_distributions(self, output_path: Path):
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
            trad_scores = self.comparison_metrics["traditional_rag"][metric]["scores"]
            hanrag_scores = self.comparison_metrics["hanrag"][metric]["scores"]

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
        plt.savefig(
            output_path / "metric_distributions.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of the evaluation."""
        if not self.comparison_metrics:
            raise ValueError("No evaluation results available. Run evaluation first.")

        summary = {
            "total_questions": len(self.evaluation_results["test_questions"]),
            "evaluation_time": self.evaluation_results["evaluation_time"],
            "key_improvements": {},
        }

        comparison = self.comparison_metrics["comparison"]
        for metric in [
            "faithfulness",
            "answer_relevance",
            "context_precision",
            "context_recall",
            "bleu_score",
        ]:
            improvement = comparison[f"{metric}_improvement"]
            summary["key_improvements"][metric] = {
                "absolute": improvement["absolute"],
                "relative_percent": improvement["relative"],
            }

        summary["processing_time_ratio"] = comparison["processing_time_ratio"]
        summary["confidence_improvement"] = comparison["confidence_improvement"]

        return summary

    def print_summary(self):
        """Print a summary of the evaluation results."""
        summary = self.get_summary_statistics()

        print("\n" + "=" * 60)
        print("RAG vs HANRAG EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total Questions Evaluated: {summary['total_questions']}")
        print(
            f"Total Evaluation Time: {summary['evaluation_time']['total']:.2f} seconds"
        )
        print()

        print("KEY IMPROVEMENTS (HANRAG vs Traditional RAG):")
        print("-" * 50)
        for metric, improvement in summary["key_improvements"].items():
            print(
                f"{metric.replace('_', ' ').title():<20}: {improvement['relative_percent']:>6.1f}%"
            )

        print(f"\nProcessing Time Ratio: {summary['processing_time_ratio']:.2f}x")
        print(
            f"Confidence Improvement: {summary['confidence_improvement']['relative']:.1f}%"
        )
        print("=" * 60)
