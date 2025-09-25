"""
Metrics for comparing traditional RAG with HANRAG.
"""

import re
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


class ComparisonMetrics:
    """
    Comprehensive metrics for comparing RAG systems.
    Implements RAGAS-style metrics and additional custom metrics.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the metrics calculator."""
        self.embedding_model = SentenceTransformer(embedding_model)
        self.smoothing = SmoothingFunction().method1

    def calculate_faithfulness(self, answer: str, context: str) -> float:
        """
        Calculate faithfulness metric - how well the answer is grounded in the context.

        Args:
            answer: Generated answer
            context: Retrieved context

        Returns:
            Faithfulness score between 0 and 1
        """
        # Tokenize and clean text
        answer_tokens = set(word_tokenize(answer.lower()))
        context_tokens = set(word_tokenize(context.lower()))

        # Remove common stopwords
        stopwords = set(nltk.corpus.stopwords.words("english"))
        answer_tokens = answer_tokens - stopwords
        context_tokens = context_tokens - stopwords

        # Calculate overlap
        if not answer_tokens:
            return 0.0

        overlap = len(answer_tokens.intersection(context_tokens))
        faithfulness = overlap / len(answer_tokens)

        return min(faithfulness, 1.0)

    def calculate_answer_relevance(self, question: str, answer: str) -> float:
        """
        Calculate answer relevance - how well the answer addresses the question.

        Args:
            question: Original question
            answer: Generated answer

        Returns:
            Relevance score between 0 and 1
        """
        # Get embeddings
        question_embedding = self.embedding_model.encode([question])
        answer_embedding = self.embedding_model.encode([answer])

        # Calculate cosine similarity
        similarity = cosine_similarity(question_embedding, answer_embedding)[0][0]

        return max(0.0, similarity)

    def calculate_context_precision(
        self, question: str, retrieved_docs: List[str]
    ) -> float:
        """
        Calculate context precision - relevance of retrieved documents to the question.

        Args:
            question: Original question
            retrieved_docs: List of retrieved document contents

        Returns:
            Context precision score between 0 and 1
        """
        if not retrieved_docs:
            return 0.0

        # Get embeddings
        question_embedding = self.embedding_model.encode([question])
        doc_embeddings = self.embedding_model.encode(retrieved_docs)

        # Calculate similarities
        similarities = cosine_similarity(question_embedding, doc_embeddings)[0]

        # Calculate precision (average relevance)
        precision = np.mean(similarities)

        return max(0.0, precision)

    def calculate_context_recall(
        self, ground_truth: str, retrieved_docs: List[str]
    ) -> float:
        """
        Calculate context recall - how much of the ground truth is covered by retrieved docs.

        Args:
            ground_truth: Expected answer or key information
            retrieved_docs: List of retrieved document contents

        Returns:
            Context recall score between 0 and 1
        """
        if not retrieved_docs:
            return 0.0

        # Combine all retrieved documents
        combined_context = " ".join(retrieved_docs)

        # Tokenize and clean
        gt_tokens = set(word_tokenize(ground_truth.lower()))
        context_tokens = set(word_tokenize(combined_context.lower()))

        # Remove stopwords
        stopwords = set(nltk.corpus.stopwords.words("english"))
        gt_tokens = gt_tokens - stopwords
        context_tokens = context_tokens - stopwords

        if not gt_tokens:
            return 1.0

        # Calculate recall
        overlap = len(gt_tokens.intersection(context_tokens))
        recall = overlap / len(gt_tokens)

        return min(recall, 1.0)

    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """
        Calculate BLEU score for answer quality.

        Args:
            reference: Reference answer
            candidate: Generated answer

        Returns:
            BLEU score between 0 and 1
        """
        reference_tokens = [word_tokenize(reference.lower())]
        candidate_tokens = word_tokenize(candidate.lower())

        try:
            bleu = sentence_bleu(
                reference_tokens, candidate_tokens, smoothing_function=self.smoothing
            )
            return bleu
        except:
            return 0.0

    def calculate_rouge_score(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores (simplified implementation).

        Args:
            reference: Reference answer
            candidate: Generated answer

        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
        """

        def _get_ngrams(text: str, n: int) -> set:
            tokens = word_tokenize(text.lower())
            return set(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

        def _get_lcs_length(seq1: list, seq2: list) -> int:
            m, n = len(seq1), len(seq2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if seq1[i - 1] == seq2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

            return dp[m][n]

        # ROUGE-1
        ref_1grams = _get_ngrams(reference, 1)
        cand_1grams = _get_ngrams(candidate, 1)
        rouge_1 = (
            len(ref_1grams.intersection(cand_1grams)) / len(ref_1grams)
            if ref_1grams
            else 0
        )

        # ROUGE-2
        ref_2grams = _get_ngrams(reference, 2)
        cand_2grams = _get_ngrams(candidate, 2)
        rouge_2 = (
            len(ref_2grams.intersection(cand_2grams)) / len(ref_2grams)
            if ref_2grams
            else 0
        )

        # ROUGE-L
        ref_tokens = word_tokenize(reference.lower())
        cand_tokens = word_tokenize(candidate.lower())
        lcs_length = _get_lcs_length(ref_tokens, cand_tokens)
        rouge_l = lcs_length / len(ref_tokens) if ref_tokens else 0

        return {"rouge_1": rouge_1, "rouge_2": rouge_2, "rouge_l": rouge_l}

    def calculate_noise_resistance(
        self,
        noisy_question: str,
        clean_question: str,
        noisy_answer: str,
        clean_answer: str,
    ) -> float:
        """
        Calculate noise resistance - how well the system handles noisy inputs.

        Args:
            noisy_question: Question with noise
            clean_question: Clean version of the question
            noisy_answer: Answer to noisy question
            clean_answer: Answer to clean question

        Returns:
            Noise resistance score between 0 and 1
        """
        # Calculate similarity between answers
        noisy_emb = self.embedding_model.encode([noisy_answer])
        clean_emb = self.embedding_model.encode([clean_answer])

        answer_similarity = cosine_similarity(noisy_emb, clean_emb)[0][0]

        return max(0.0, answer_similarity)

    def calculate_multi_hop_accuracy(
        self, reasoning_steps: List[Dict], expected_steps: List[str]
    ) -> float:
        """
        Calculate multi-hop reasoning accuracy.

        Args:
            reasoning_steps: List of reasoning steps from HANRAG
            expected_steps: List of expected reasoning steps

        Returns:
            Multi-hop accuracy score between 0 and 1
        """
        if not reasoning_steps or not expected_steps:
            return 0.0

        # Extract reasoning text from steps
        actual_reasoning = [step.get("reasoning", "") for step in reasoning_steps]

        # Calculate average similarity with expected steps
        similarities = []
        for actual, expected in zip(actual_reasoning, expected_steps):
            if actual and expected:
                actual_emb = self.embedding_model.encode([actual])
                expected_emb = self.embedding_model.encode([expected])
                sim = cosine_similarity(actual_emb, expected_emb)[0][0]
                similarities.append(sim)

        return np.mean(similarities) if similarities else 0.0

    def calculate_comprehensive_metrics(
        self,
        traditional_results: List[Dict],
        hanrag_results: List[Dict],
        ground_truths: List[str],
        questions: List[str],
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive comparison metrics.

        Args:
            traditional_results: Results from traditional RAG
            hanrag_results: Results from HANRAG
            ground_truths: Ground truth answers
            questions: Original questions

        Returns:
            Dictionary with all comparison metrics
        """
        metrics = {"traditional_rag": {}, "hanrag": {}, "comparison": {}}

        # Calculate metrics for each system
        for system_name, results in [
            ("traditional_rag", traditional_results),
            ("hanrag", hanrag_results),
        ]:

            faithfulness_scores = []
            answer_relevance_scores = []
            context_precision_scores = []
            context_recall_scores = []
            bleu_scores = []
            rouge_scores = []
            processing_times = []
            confidences = []

            for i, (result, gt, question) in enumerate(
                zip(results, ground_truths, questions)
            ):
                # Handle both dict and HANRAGResponse objects
                if hasattr(result, "retrieved_documents"):
                    # HANRAGResponse object
                    retrieved_docs = result.retrieved_documents
                    answer = result.answer
                    confidence = result.confidence
                    processing_time = result.processing_time
                else:
                    # Dictionary format
                    retrieved_docs = result.get("retrieved_documents", [])
                    answer = result.get("answer", "")
                    confidence = result.get("confidence", 0.0)
                    processing_time = result.get("processing_time", 0.0)

                # Extract context from retrieved documents
                if hasattr(retrieved_docs[0], "content") if retrieved_docs else False:
                    # RetrievalResult objects
                    context = " ".join([doc.content for doc in retrieved_docs])
                    doc_contents = [doc.content for doc in retrieved_docs]
                else:
                    # Dictionary format
                    context = " ".join(
                        [doc.get("content", "") for doc in retrieved_docs]
                    )
                    doc_contents = [doc.get("content", "") for doc in retrieved_docs]

                # Calculate metrics
                faithfulness = self.calculate_faithfulness(answer, context)
                answer_relevance = self.calculate_answer_relevance(question, answer)
                context_precision = self.calculate_context_precision(
                    question, doc_contents
                )
                context_recall = self.calculate_context_recall(gt, [context])
                bleu = self.calculate_bleu_score(gt, answer)
                rouge = self.calculate_rouge_score(gt, answer)

                faithfulness_scores.append(faithfulness)
                answer_relevance_scores.append(answer_relevance)
                context_precision_scores.append(context_precision)
                context_recall_scores.append(context_recall)
                bleu_scores.append(bleu)
                rouge_scores.append(rouge)
                processing_times.append(processing_time)
                confidences.append(confidence)

            # Store aggregated metrics
            metrics[system_name] = {
                "faithfulness": {
                    "mean": np.mean(faithfulness_scores),
                    "std": np.std(faithfulness_scores),
                    "scores": faithfulness_scores,
                },
                "answer_relevance": {
                    "mean": np.mean(answer_relevance_scores),
                    "std": np.std(answer_relevance_scores),
                    "scores": answer_relevance_scores,
                },
                "context_precision": {
                    "mean": np.mean(context_precision_scores),
                    "std": np.std(context_precision_scores),
                    "scores": context_precision_scores,
                },
                "context_recall": {
                    "mean": np.mean(context_recall_scores),
                    "std": np.std(context_recall_scores),
                    "scores": context_recall_scores,
                },
                "bleu_score": {
                    "mean": np.mean(bleu_scores),
                    "std": np.std(bleu_scores),
                    "scores": bleu_scores,
                },
                "rouge_scores": {
                    "rouge_1": np.mean([r["rouge_1"] for r in rouge_scores]),
                    "rouge_2": np.mean([r["rouge_2"] for r in rouge_scores]),
                    "rouge_l": np.mean([r["rouge_l"] for r in rouge_scores]),
                },
                "processing_time": {
                    "mean": np.mean(processing_times),
                    "std": np.std(processing_times),
                    "scores": processing_times,
                },
                "confidence": {
                    "mean": np.mean(confidences),
                    "std": np.std(confidences),
                    "scores": confidences,
                },
            }

        # Calculate comparison metrics
        comparison = {}
        for metric in [
            "faithfulness",
            "answer_relevance",
            "context_precision",
            "context_recall",
            "bleu_score",
        ]:
            trad_mean = metrics["traditional_rag"][metric]["mean"]
            hanrag_mean = metrics["hanrag"][metric]["mean"]

            comparison[f"{metric}_improvement"] = {
                "absolute": hanrag_mean - trad_mean,
                "relative": (
                    ((hanrag_mean - trad_mean) / trad_mean * 100)
                    if trad_mean > 0
                    else 0
                ),
            }

        # Processing time comparison
        trad_time = metrics["traditional_rag"]["processing_time"]["mean"]
        hanrag_time = metrics["hanrag"]["processing_time"]["mean"]
        comparison["processing_time_ratio"] = (
            hanrag_time / trad_time if trad_time > 0 else 0
        )

        # Confidence comparison
        trad_conf = metrics["traditional_rag"]["confidence"]["mean"]
        hanrag_conf = metrics["hanrag"]["confidence"]["mean"]
        comparison["confidence_improvement"] = {
            "absolute": hanrag_conf - trad_conf,
            "relative": (
                ((hanrag_conf - trad_conf) / trad_conf * 100) if trad_conf > 0 else 0
            ),
        }

        metrics["comparison"] = comparison

        return metrics
