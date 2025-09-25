"""
Generation component for HANRAG system.
Implements accurate answer generation with confidence estimation.
"""

import time
from typing import List, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .models import (
    RetrievalResult,
    ReasoningStep,
    MultiHopQuery,
    HANRAGResponse,
    NoiseDetectionResult,
)
from .config import config


class AnswerGenerator:
    """Generates accurate answers with confidence estimation."""

    def __init__(self, model_name: Optional[str] = None):
        """Initialize the answer generator."""
        self.model_name = model_name or config.default_model
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=0.1,  # Low temperature for consistency
            max_tokens=1000,
        )
        self._setup_prompts()

    def _setup_prompts(self):
        """Setup prompt templates for different tasks."""
        self.reasoning_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at multi-hop reasoning. Given a question and retrieved documents, 
            break down the reasoning into clear steps and provide an intermediate answer for each step.
            
            Guidelines:
            1. Identify the key information needed to answer the question
            2. Break down complex questions into simpler sub-questions
            3. Use information from the retrieved documents
            4. Provide clear reasoning for each step
            5. Estimate your confidence in each step (0.0 to 1.0)
            
            Format your response as:
            Step 1: [Sub-question]
            Reasoning: [Your reasoning process]
            Answer: [Intermediate answer]
            Confidence: [0.0-1.0]
            
            Step 2: [Next sub-question]
            Reasoning: [Your reasoning process]
            Answer: [Intermediate answer]
            Confidence: [0.0-1.0]
            
            ...and so on.""",
                ),
                (
                    "human",
                    "Question: {question}\n\nRetrieved Documents:\n{documents}\n\nPlease provide step-by-step reasoning.",
                ),
            ]
        )

        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at providing accurate, well-reasoned answers. 
            Based on the reasoning steps and retrieved documents, provide a final answer.
            
            Guidelines:
            1. Synthesize information from all reasoning steps
            2. Use evidence from the retrieved documents
            3. Be precise and factual
            4. If uncertain, express your uncertainty
            5. Provide a confidence score (0.0 to 1.0)
            
            Format your response as:
            Final Answer: [Your final answer]
            Confidence: [0.0-1.0]
            Evidence: [Key evidence supporting your answer]""",
                ),
                (
                    "human",
                    "Question: {question}\n\nReasoning Steps:\n{reasoning_steps}\n\nRetrieved Documents:\n{documents}\n\nPlease provide the final answer.",
                ),
            ]
        )

        self.confidence_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at evaluating answer confidence. 
            Given a question, answer, and supporting evidence, estimate the confidence level.
            
            Consider:
            1. How well the evidence supports the answer
            2. The quality and relevance of the retrieved documents
            3. The clarity and specificity of the question
            4. Any potential ambiguities or uncertainties
            
            Provide a confidence score from 0.0 (very uncertain) to 1.0 (very confident).""",
                ),
                (
                    "human",
                    "Question: {question}\nAnswer: {answer}\nEvidence: {evidence}\n\nWhat is your confidence in this answer?",
                ),
            ]
        )

    def generate_multi_hop_answer(
        self, query: MultiHopQuery, retrieved_documents: List[RetrievalResult]
    ) -> HANRAGResponse:
        """Generate a multi-hop answer with reasoning steps."""
        start_time = time.time()

        # Step 1: Generate reasoning steps
        reasoning_steps = self._generate_reasoning_steps(
            query.original_question, retrieved_documents
        )

        # Step 2: Generate final answer
        final_answer, final_confidence = self._generate_final_answer(
            query.original_question, reasoning_steps, retrieved_documents
        )

        # Step 3: Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            reasoning_steps, final_confidence
        )

        processing_time = time.time() - start_time

        # Update the query with results
        query.reasoning_steps = reasoning_steps
        query.final_answer = final_answer
        query.confidence = overall_confidence

        return HANRAGResponse(
            query=query,
            answer=final_answer,
            confidence=overall_confidence,
            reasoning_chain=reasoning_steps,
            retrieved_documents=retrieved_documents,
            processing_time=processing_time,
            metadata={
                "model_used": self.model_name,
                "num_reasoning_steps": len(reasoning_steps),
                "num_retrieved_docs": len(retrieved_documents),
            },
        )

    def _generate_reasoning_steps(
        self, question: str, documents: List[RetrievalResult]
    ) -> List[ReasoningStep]:
        """Generate reasoning steps for multi-hop question answering."""
        # Format documents for the prompt
        doc_text = "\n\n".join(
            [
                f"Document {i+1} (Score: {doc.score:.3f}):\n{doc.content}"
                for i, doc in enumerate(documents)
            ]
        )

        # Generate reasoning using the LLM
        messages = self.reasoning_prompt.format_messages(
            question=question, documents=doc_text
        )

        response = self.llm.invoke(messages)
        reasoning_text = response.content

        # Parse the reasoning steps
        steps = self._parse_reasoning_steps(reasoning_text, documents)

        return steps

    def _parse_reasoning_steps(
        self, reasoning_text: str, documents: List[RetrievalResult]
    ) -> List[ReasoningStep]:
        """Parse reasoning steps from LLM response."""
        steps = []
        lines = reasoning_text.split("\n")

        current_step = None
        step_number = 1

        for line in lines:
            line = line.strip()

            if line.startswith("Step") and ":" in line:
                # Save previous step if exists
                if current_step:
                    steps.append(current_step)

                # Start new step
                question = line.split(":", 1)[1].strip()
                current_step = ReasoningStep(
                    step_number=step_number,
                    question=question,
                    retrieved_documents=documents,
                    reasoning="",
                    confidence=0.0,
                )
                step_number += 1

            elif line.startswith("Reasoning:") and current_step:
                current_step.reasoning = line.split(":", 1)[1].strip()

            elif line.startswith("Answer:") and current_step:
                current_step.intermediate_answer = line.split(":", 1)[1].strip()

            elif line.startswith("Confidence:") and current_step:
                try:
                    confidence_str = line.split(":", 1)[1].strip()
                    current_step.confidence = float(confidence_str)
                except ValueError:
                    current_step.confidence = 0.5  # Default confidence

        # Add the last step
        if current_step:
            steps.append(current_step)

        return steps

    def _generate_final_answer(
        self,
        question: str,
        reasoning_steps: List[ReasoningStep],
        documents: List[RetrievalResult],
    ) -> Tuple[str, float]:
        """Generate the final answer based on reasoning steps."""
        # Format reasoning steps for the prompt
        reasoning_text = "\n\n".join(
            [
                f"Step {step.step_number}: {step.question}\n"
                f"Reasoning: {step.reasoning}\n"
                f"Answer: {step.intermediate_answer}\n"
                f"Confidence: {step.confidence}"
                for step in reasoning_steps
            ]
        )

        # Format documents
        doc_text = "\n\n".join(
            [
                f"Document {i+1} (Score: {doc.score:.3f}):\n{doc.content}"
                for i, doc in enumerate(documents)
            ]
        )

        # Generate final answer
        messages = self.answer_prompt.format_messages(
            question=question, reasoning_steps=reasoning_text, documents=doc_text
        )

        response = self.llm.invoke(messages)
        answer_text = response.content

        # Parse final answer and confidence
        final_answer, confidence = self._parse_final_answer(answer_text)

        return final_answer, confidence

    def _parse_final_answer(self, answer_text: str) -> Tuple[str, float]:
        """Parse final answer and confidence from LLM response."""
        lines = answer_text.split("\n")
        final_answer = ""
        confidence = 0.5  # Default confidence

        for line in lines:
            line = line.strip()
            if line.startswith("Final Answer:"):
                final_answer = line.split(":", 1)[1].strip()
            elif line.startswith("Confidence:"):
                try:
                    confidence_str = line.split(":", 1)[1].strip()
                    confidence = float(confidence_str)
                except ValueError:
                    confidence = 0.5

        return final_answer, confidence

    def _calculate_overall_confidence(
        self, reasoning_steps: List[ReasoningStep], final_confidence: float
    ) -> float:
        """Calculate overall confidence based on reasoning steps and final answer."""
        if not reasoning_steps:
            return final_confidence

        # Average confidence from reasoning steps
        step_confidences = [step.confidence for step in reasoning_steps]
        avg_step_confidence = sum(step_confidences) / len(step_confidences)

        # Weighted combination of step confidence and final confidence
        overall_confidence = (avg_step_confidence * 0.6) + (final_confidence * 0.4)

        return min(1.0, max(0.0, overall_confidence))


class NoiseResistantGenerator(AnswerGenerator):
    """Noise-resistant version of the answer generator."""

    def __init__(self, model_name: Optional[str] = None):
        super().__init__(model_name)
        self.noise_threshold = config.noise_threshold

    def generate_with_noise_detection(
        self,
        query: MultiHopQuery,
        retrieved_documents: List[RetrievalResult],
        is_query_noisy: bool = False,
    ) -> HANRAGResponse:
        """Generate answer with noise detection and resistance."""
        if is_query_noisy:
            # Apply noise-resistant strategies
            query = self._clarify_noisy_query(query)
            retrieved_documents = self._filter_noisy_documents(retrieved_documents)

        # Generate answer using parent method
        response = self.generate_multi_hop_answer(query, retrieved_documents)

        # Detect noise in the generated answer
        noise_detection = self._detect_answer_noise(
            query.original_question, response.answer, retrieved_documents
        )

        # Update response with noise information
        response.metadata["noise_detection"] = noise_detection.dict()
        query.noise_detected = noise_detection.is_noisy
        query.noise_level = noise_detection.noise_score

        return response

    def _clarify_noisy_query(self, query: MultiHopQuery) -> MultiHopQuery:
        """Clarify noisy queries by generating more specific versions."""
        clarification_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at clarifying ambiguous questions. 
            Given a potentially noisy or ambiguous question, generate a clearer, more specific version.
            
            Guidelines:
            1. Identify ambiguous terms and replace them with specific ones
            2. Add context where needed
            3. Break down complex questions into clearer parts
            4. Maintain the original intent of the question
            
            Provide the clarified question.""",
                ),
                (
                    "human",
                    "Original question: {question}\n\nPlease provide a clearer version of this question.",
                ),
            ]
        )

        messages = clarification_prompt.format_messages(
            question=query.original_question
        )
        response = self.llm.invoke(messages)
        clarified_question = response.content.strip()

        # Update the query with clarified question
        query.original_question = clarified_question
        return query

    def _filter_noisy_documents(
        self, documents: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Filter out potentially noisy or irrelevant documents."""
        filtered_docs = []

        for doc in documents:
            # Check document quality indicators
            is_high_quality = (
                doc.score > config.similarity_threshold
                and len(doc.content) > 50  # Not too short
                and len(doc.content) < 5000  # Not too long
                and not self._contains_noise_indicators(doc.content)
            )

            if is_high_quality:
                filtered_docs.append(doc)

        return filtered_docs

    def _contains_noise_indicators(self, content: str) -> bool:
        """Check if content contains noise indicators."""
        noise_indicators = [
            "error",
            "broken",
            "incomplete",
            "placeholder",
            "lorem ipsum",
            "test content",
            "sample text",
        ]

        content_lower = content.lower()
        return any(indicator in content_lower for indicator in noise_indicators)

    def _detect_answer_noise(
        self, question: str, answer: str, documents: List[RetrievalResult]
    ) -> NoiseDetectionResult:
        """Detect noise in the generated answer using heuristic-based approach."""

        # Heuristic 1: Check for explicit uncertainty patterns
        uncertainty_patterns = [
            "I don't know",
            "I'm not sure",
            "I cannot answer",
            "insufficient information",
            "not enough data",
            "unclear",
            "ambiguous",
            "uncertain",
            "hard to say",
            "difficult to determine",
        ]

        answer_lower = answer.lower()
        uncertainty_count = sum(
            1 for pattern in uncertainty_patterns if pattern in answer_lower
        )
        uncertainty_score = min(1.0, uncertainty_count * 0.4)

        # Heuristic 2: Check answer relevance to question
        relevance_score = self._calculate_question_answer_relevance(question, answer)

        # Heuristic 3: Check document support (how well answer is grounded in retrieved docs)
        support_score = self._calculate_document_support(answer, documents)

        # Heuristic 4: Check answer completeness and specificity
        completeness_score = self._calculate_answer_completeness(question, answer)

        # Heuristic 5: Check for repetition or circular reasoning
        repetition_score = self._calculate_repetition_score(answer)

        # Weighted combination of heuristics (as per HANRAG's heuristic approach)
        noise_score = (
            uncertainty_score * 0.3  # Explicit uncertainty
            + (1 - relevance_score) * 0.25  # Low relevance to question
            + (1 - support_score) * 0.25  # Poor document support
            + (1 - completeness_score) * 0.15  # Incomplete answer
            + repetition_score * 0.05  # Repetitive content
        )

        noise_score = min(1.0, max(0.0, noise_score))
        is_noisy = noise_score > self.noise_threshold

        return NoiseDetectionResult(
            is_noisy=is_noisy,
            noise_score=noise_score,
            noise_type="answer_quality" if is_noisy else None,
            confidence=0.8,
            explanation=f"Answer noise score: {noise_score:.3f} (uncertainty: {uncertainty_score:.2f}, relevance: {relevance_score:.2f}, support: {support_score:.2f}, completeness: {completeness_score:.2f})",
        )

    def _calculate_document_support(
        self, answer: str, documents: List[RetrievalResult]
    ) -> float:
        """Calculate how well the answer is supported by the documents."""
        if not documents:
            return 0.0

        # Simple keyword overlap between answer and documents
        answer_words = set(answer.lower().split())
        total_overlap = 0

        for doc in documents:
            doc_words = set(doc.content.lower().split())
            overlap = len(answer_words.intersection(doc_words))
            total_overlap += overlap

        # Normalize by number of documents and answer length
        avg_overlap = total_overlap / len(documents)
        support_score = min(1.0, avg_overlap / max(1, len(answer_words) * 0.3))

        return support_score

    def _calculate_question_answer_relevance(self, question: str, answer: str) -> float:
        """Calculate how relevant the answer is to the question using keyword overlap."""
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())

        # Remove common stop words
        stop_words = {
            "what",
            "how",
            "when",
            "where",
            "why",
            "who",
            "which",
            "is",
            "are",
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
        }
        question_words = question_words - stop_words
        answer_words = answer_words - stop_words

        if not question_words:
            return 1.0  # If no meaningful words in question, assume relevant

        overlap = len(question_words.intersection(answer_words))
        relevance = overlap / len(question_words)
        return min(1.0, relevance)

    def _calculate_answer_completeness(self, question: str, answer: str) -> float:
        """Calculate how complete and specific the answer is."""
        # Check for question words that suggest what kind of answer is expected
        question_lower = question.lower()

        # Length-based completeness (answers should be substantial)
        length_score = min(1.0, len(answer.split()) / 10)  # Expect at least 10 words

        # Check for specific information patterns
        specificity_score = 0.0
        if any(word in question_lower for word in ["what", "define", "explain"]):
            # Should have definitions or explanations
            if any(
                word in answer.lower()
                for word in ["is", "are", "refers to", "means", "defined as"]
            ):
                specificity_score += 0.5
        elif any(word in question_lower for word in ["how", "why"]):
            # Should have process or reasoning
            if any(
                word in answer.lower()
                for word in ["because", "by", "through", "using", "due to"]
            ):
                specificity_score += 0.5
        elif any(word in question_lower for word in ["when", "where", "who"]):
            # Should have specific facts
            if len(answer.split()) > 5:  # Substantial factual answer
                specificity_score += 0.5
        else:
            # General question, any substantial answer is good
            specificity_score = 0.5

        return (length_score * 0.6) + (specificity_score * 0.4)

    def _calculate_repetition_score(self, answer: str) -> float:
        """Calculate how repetitive the answer is."""
        words = answer.lower().split()
        if len(words) < 3:
            return 0.0

        # Check for repeated phrases (3+ words)
        phrases = {}
        for i in range(len(words) - 2):
            phrase = " ".join(words[i : i + 3])
            phrases[phrase] = phrases.get(phrase, 0) + 1

        # Calculate repetition ratio
        total_phrases = len(words) - 2
        repeated_phrases = sum(count - 1 for count in phrases.values() if count > 1)

        if total_phrases == 0:
            return 0.0

        repetition_ratio = repeated_phrases / total_phrases
        return min(1.0, repetition_ratio * 2)  # Scale up for more sensitive detection
