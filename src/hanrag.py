"""
Main HANRAG system implementation.
Integrates retrieval, generation, and noise resistance components.
"""

import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.callbacks.manager import get_openai_callback

from .models import (
    MultiHopQuery,
    HANRAGResponse,
    RetrievalResult,
    ReasoningStep,
    NoiseDetectionResult,
)
from .retrieval import NoiseResistantRetriever
from .generation import NoiseResistantGenerator
from .revelator import Revelator
from .models import QueryType
from .config import config


class HANRAGSystem:
    """
    HANRAG: Heuristic Accurate Noise-resistant Retrieval-Augmented Generation
    for Multi-hop Question Answering
    """

    def __init__(
        self, model_name: Optional[str] = None, embedding_model: Optional[str] = None
    ):
        """Initialize the HANRAG system."""
        self.retriever = NoiseResistantRetriever(embedding_model)
        self.generator = NoiseResistantGenerator(model_name)
        self.revelator = Revelator(model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        self.max_hops = config.max_hops

    def add_knowledge_base(self, documents: List[Document]):
        """Add documents to the knowledge base."""
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)

        # Add to retriever
        self.retriever.add_documents(chunks)

    def answer_question(
        self, question: str, max_hops: Optional[int] = None
    ) -> HANRAGResponse:
        """
        Answer a question using the HANRAG framework with Revelator routing.

        Args:
            question: The question to answer
            max_hops: Maximum number of reasoning hops (default: config.max_hops)

        Returns:
            HANRAGResponse with answer, reasoning, and metadata
        """
        max_hops = max_hops or self.max_hops
        start_time = time.time()

        # Use LangSmith tracing for the entire question answering process
        with get_openai_callback() as cb:
            # Step 1: Route the query using Revelator
            query_type = self.revelator.route_query(question)

            # Step 2: Process based on query type
            if query_type == QueryType.STRAIGHTFORWARD:
                response = self._handle_straightforward_query(question)
            elif query_type == QueryType.SINGLE_STEP:
                response = self._handle_single_step_query(question)
            elif query_type == QueryType.COMPOUND:
                response = self._handle_compound_query(question)
            elif query_type == QueryType.COMPLEX:
                response = self._handle_complex_query(question, max_hops)
            else:
                # Fallback to single-step
                response = self._handle_single_step_query(question)

            # Update processing time
            response.processing_time = time.time() - start_time
            response.metadata["query_type"] = query_type.value

            # Add token usage information to metadata
            response.metadata.update(
                {
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost": cb.total_cost,
                }
            )

        return response

    def _handle_straightforward_query(self, question: str) -> HANRAGResponse:
        """Handle straightforward queries that don't require retrieval."""
        query = MultiHopQuery(original_question=question)

        # Generate answer directly using LLM
        response = self.generator.generate_with_noise_detection(query, [], False)

        return response

    def _handle_single_step_query(self, question: str) -> HANRAGResponse:
        """Handle single-step queries using ANRAG."""
        query = MultiHopQuery(original_question=question)

        # Retrieve documents with noise detection
        docs, is_noisy, noise_score = self.retriever.retrieve_with_noise_detection(
            question, k=config.top_k_documents
        )

        # Filter relevant documents using Revelator
        relevant_docs = self.revelator.filter_relevant_documents(question, docs)

        query.noise_detected = is_noisy
        query.noise_level = noise_score

        # Generate answer
        response = self.generator.generate_with_noise_detection(
            query, relevant_docs, is_noisy
        )

        return response

    def _handle_compound_query(self, question: str) -> HANRAGResponse:
        """Handle compound queries with parallel processing."""
        query = MultiHopQuery(original_question=question)

        # Decompose the query
        sub_queries = self.revelator.decompose_query(question)

        # Process sub-queries in parallel
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            sub_results = loop.run_until_complete(
                self.revelator.process_compound_query_parallel(
                    sub_queries, self.retriever, self.generator
                )
            )
        finally:
            loop.close()

        # Combine answers
        final_answer = self.revelator.combine_compound_answers(question, sub_results)

        # Create response
        response = HANRAGResponse(
            query=query,
            answer=final_answer,
            confidence=sum(r["confidence"] for r in sub_results) / len(sub_results),
            reasoning_chain=[],
            retrieved_documents=[],
            processing_time=0.0,
            metadata={
                "sub_queries": sub_queries,
                "sub_results": sub_results,
                "processing_type": "parallel",
            },
        )

        return response

    def _handle_complex_query(self, question: str, max_hops: int) -> HANRAGResponse:
        """Handle complex queries with iterative reasoning."""
        query = MultiHopQuery(original_question=question)
        reasoning_steps = []

        # Iterative reasoning process
        for hop in range(max_hops):
            # Refine seed question
            if hop == 0:
                seed_question = self.revelator.refine_seed_question(question)
            else:
                seed_question = self.revelator.refine_seed_question(
                    question, reasoning_steps
                )

            # Check if sufficient
            if isinstance(seed_question, bool) and seed_question:
                break

            # Retrieve documents for seed question
            docs, is_noisy, noise_score = self.retriever.retrieve_with_noise_detection(
                seed_question, k=config.top_k_documents
            )

            # Filter relevant documents
            relevant_docs = self.revelator.filter_relevant_documents(
                seed_question, docs
            )

            # Generate reasoning step
            step = self._generate_reasoning_step(seed_question, relevant_docs, hop + 1)
            reasoning_steps.append(step)

            # Check if we should end reasoning
            if self.revelator.should_end_reasoning(question, reasoning_steps):
                break

        # Generate final answer
        all_docs = []
        for step in reasoning_steps:
            all_docs.extend(step.retrieved_documents)

        # Remove duplicates
        unique_docs = self._merge_documents(all_docs, [])

        response = self.generator.generate_with_noise_detection(
            query, unique_docs, False
        )

        # Update response with reasoning steps
        response.reasoning_chain = reasoning_steps
        response.query.reasoning_steps = reasoning_steps

        return response

    def _generate_reasoning_step(
        self, question: str, documents: List[RetrievalResult], step_number: int
    ) -> ReasoningStep:
        """Generate a single reasoning step."""
        # Use the generator to create a reasoning step
        reasoning_steps = self.generator._generate_reasoning_steps(question, documents)

        if reasoning_steps:
            # Take the first step and update step number
            step = reasoning_steps[0]
            step.step_number = step_number
            return step
        else:
            # Fallback if no reasoning steps generated
            return ReasoningStep(
                step_number=step_number,
                question=question,
                retrieved_documents=documents,
                reasoning="Unable to generate reasoning step",
                confidence=0.0,
            )

    def _retrieve_follow_up_documents(
        self, reasoning_step: ReasoningStep, current_docs: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Retrieve additional documents based on reasoning step."""
        # Extract key terms from reasoning step
        key_terms = self._extract_key_terms(reasoning_step)

        # Create follow-up queries
        follow_up_queries = self._create_follow_up_queries(key_terms)

        # Retrieve documents for each follow-up query
        follow_up_docs = []
        for query in follow_up_queries:
            docs, _, _ = self.retriever.retrieve_with_noise_detection(query, k=3)
            follow_up_docs.extend(docs)

        return follow_up_docs

    def _extract_key_terms(self, reasoning_step: ReasoningStep) -> List[str]:
        """Extract key terms from reasoning step."""
        # Simple key term extraction - in practice, you'd use NLP techniques
        text = f"{reasoning_step.question} {reasoning_step.reasoning} {reasoning_step.intermediate_answer}"

        # Remove common words and extract meaningful terms
        common_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
        }

        words = text.lower().split()
        # Strip punctuation from words
        import string

        key_terms = []
        for word in words:
            clean_word = word.strip(string.punctuation)
            if clean_word not in common_words and len(clean_word) > 3:
                key_terms.append(clean_word)

        unique_terms = list(set(key_terms))
        return unique_terms[:10]  # Return top 10 unique terms

    def _create_follow_up_queries(self, key_terms: List[str]) -> List[str]:
        """Create follow-up queries based on key terms."""
        queries = []

        # Create queries for individual terms
        for term in key_terms:
            queries.append(f"What is {term}?")
            queries.append(f"How does {term} work?")

        # Create combined queries
        if len(key_terms) >= 2:
            queries.append(f"How are {key_terms[0]} and {key_terms[1]} related?")

        return queries[:3]  # Limit to 3 follow-up queries

    def _merge_documents(
        self, docs1: List[RetrievalResult], docs2: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Merge two lists of documents, removing duplicates."""
        # Create a dictionary to track unique documents
        unique_docs = {}

        # Add documents from first list
        for doc in docs1:
            unique_docs[doc.document_id] = doc

        # Add documents from second list (keep higher score if duplicate)
        for doc in docs2:
            if doc.document_id in unique_docs:
                if doc.score > unique_docs[doc.document_id].score:
                    unique_docs[doc.document_id] = doc
            else:
                unique_docs[doc.document_id] = doc

        # Sort by score and return top documents
        merged_docs = list(unique_docs.values())
        merged_docs.sort(key=lambda x: x.score, reverse=True)

        return merged_docs[: config.top_k_documents]

    def batch_answer_questions(
        self, questions: List[str], max_hops: Optional[int] = None
    ) -> List[HANRAGResponse]:
        """Answer multiple questions in batch."""
        responses = []

        for question in questions:
            try:
                response = self.answer_question(question, max_hops)
                responses.append(response)
            except Exception as e:
                # Create error response
                error_response = HANRAGResponse(
                    query=MultiHopQuery(original_question=question),
                    answer=f"Error processing question: {str(e)}",
                    confidence=0.0,
                    reasoning_chain=[],
                    retrieved_documents=[],
                    processing_time=0.0,
                    metadata={"error": str(e)},
                )
                responses.append(error_response)

        return responses

    def evaluate_system(
        self, test_questions: List[str], expected_answers: List[str]
    ) -> Dict[str, Any]:
        """Evaluate the system performance on test questions."""
        if len(test_questions) != len(expected_answers):
            raise ValueError("Number of questions and expected answers must match")

        responses = self.batch_answer_questions(test_questions)

        # Calculate metrics
        total_questions = len(test_questions)
        high_confidence_answers = sum(1 for r in responses if r.confidence > 0.7)
        noisy_queries_detected = sum(1 for r in responses if r.query.noise_detected)

        avg_confidence = sum(r.confidence for r in responses) / total_questions
        avg_processing_time = (
            sum(r.processing_time for r in responses) / total_questions
        )
        avg_reasoning_steps = (
            sum(len(r.reasoning_chain) for r in responses) / total_questions
        )

        # Simple accuracy calculation (in practice, you'd use more sophisticated metrics)
        correct_answers = 0
        for i, (response, expected) in enumerate(zip(responses, expected_answers)):
            # Simple keyword overlap for accuracy
            answer_words = set(response.answer.lower().split())
            expected_words = set(expected.lower().split())
            overlap = len(answer_words.intersection(expected_words))
            if overlap > len(expected_words) * 0.5:  # 50% overlap threshold
                correct_answers += 1

        accuracy = correct_answers / total_questions

        return {
            "total_questions": total_questions,
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "high_confidence_rate": high_confidence_answers / total_questions,
            "noise_detection_rate": noisy_queries_detected / total_questions,
            "avg_processing_time": avg_processing_time,
            "avg_reasoning_steps": avg_reasoning_steps,
            "responses": responses,
        }
