"""
Revelator: The master agent for HANRAG system.
Implements query routing, decomposition, refinement, and discrimination.
"""

import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate

from .models import RetrievalResult, ReasoningStep, MultiHopQuery, QueryType
from .config import config


class Revelator:
    """
    The master agent that orchestrates the entire HANRAG framework.
    Implements all five core components: Router, Decomposer, Refiner,
    Relevance Discriminator, and Ending Discriminator.
    """

    def __init__(self, model_name: Optional[str] = None):
        """Initialize the Revelator with all components."""
        self.model_name = model_name or config.default_model
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=0.1,
            max_tokens=1000,
        )
        self._setup_prompts()

    def _setup_prompts(self):
        """Setup prompt templates for all Revelator components."""

        # Router prompt
        self.router_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at classifying questions into different types for a RAG system.

Classify the given question into one of these four types:

1. STRAIGHTFORWARD: Questions that can be answered directly without external knowledge retrieval (e.g., "Who is the first President of America?", "What is 2+2?")

2. SINGLE_STEP: Questions that require one retrieval step to find the answer (e.g., "When was Pan Jianwei born?", "What is the capital of France?")

3. COMPOUND: Questions that ask about multiple independent aspects of a single entity (e.g., "When was Liu Xiang born and when did he retire?", "What are the main features and price of iPhone 15?")

4. COMPLEX: Questions that require multi-step reasoning with dependent sub-questions (e.g., "Who succeeded the first President of Namibia?", "What city is the person who broadened the doctrine of philosophy of language from?")

Respond with only the classification type: STRAIGHTFORWARD, SINGLE_STEP, COMPOUND, or COMPLEX""",
                ),
                ("human", "Question: {question}\n\nClassification:"),
            ]
        )

        # Decomposer prompt
        self.decomposer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at decomposing compound questions into independent sub-questions.

Given a compound question, break it down into multiple independent sub-questions that can be answered separately and then combined.

Rules:
1. Each sub-question should be independent and answerable on its own
2. All sub-questions should relate to the same main entity or topic
3. The combination of answers should fully address the original question
4. Return the sub-questions as a JSON list

Example:
Question: "When was Liu Xiang born and when did he retire?"
Sub-questions: ["When was Liu Xiang born?", "When did Liu Xiang retire?"]

Respond with a JSON list of sub-questions:""",
                ),
                ("human", "Compound Question: {question}\n\nSub-questions:"),
            ]
        )

        # Refiner prompt
        self.refiner_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at refining complex questions into seed questions for multi-step reasoning.

Given a complex question and the current reasoning context, identify the next seed question that needs to be answered to make progress toward solving the original question.

Rules:
1. The seed question should be the next logical step in the reasoning chain
2. It should be answerable with a single retrieval step
3. The answer should help answer the original complex question
4. If this is the first step, identify the initial seed question
5. If sufficient information has been gathered, respond with "SUFFICIENT"

Context format:
- Previous steps: [list of previous reasoning steps with their answers]
- Current question: [the original complex question]

Respond with either the next seed question or "SUFFICIENT":""",
                ),
                (
                    "human",
                    """Original Question: {original_question}

Previous Steps: {previous_steps}

Next seed question or SUFFICIENT:""",
                ),
            ]
        )

        # Relevance discriminator prompt
        self.relevance_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at determining document relevance for question answering.

Given a question and a document, determine if the document is relevant for answering the question.

A document is relevant if:
1. It contains information that directly answers the question
2. It provides context or background information needed to understand the answer
3. It contains facts that support or relate to the answer

A document is NOT relevant if:
1. It's completely unrelated to the question topic
2. It's about a different entity or subject
3. It contains no useful information for the question

Respond with only "RELEVANT" or "NOT_RELEVANT":""",
                ),
                (
                    "human",
                    """Question: {question}

Document: {document}

Relevance:""",
                ),
            ]
        )

        # Ending discriminator prompt
        self.ending_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at determining when sufficient information has been gathered to answer a complex question.

Given the original question and the reasoning steps completed so far, determine if enough information has been gathered to provide a complete answer.

Consider:
1. Has the original question been fully addressed?
2. Are all necessary sub-questions answered?
3. Is there sufficient evidence to support the final answer?
4. Would additional retrieval steps add meaningful information?

Respond with only "CONTINUE" or "SUFFICIENT":""",
                ),
                (
                    "human",
                    """Original Question: {original_question}

Completed Reasoning Steps: {reasoning_steps}

Decision:""",
                ),
            ]
        )

    def route_query(self, query: str) -> QueryType:
        """
        Route a query to determine its type.

        Args:
            query: The input question

        Returns:
            QueryType: The classified query type
        """
        try:
            messages = self.router_prompt.format_messages(question=query)
            response = self.llm.invoke(messages)
            classification = response.content.strip().upper()

            # Map to enum
            if "STRAIGHTFORWARD" in classification:
                return QueryType.STRAIGHTFORWARD
            elif "SINGLE_STEP" in classification:
                return QueryType.SINGLE_STEP
            elif "COMPOUND" in classification:
                return QueryType.COMPOUND
            elif "COMPLEX" in classification:
                return QueryType.COMPLEX
            else:
                # Default to single-step if unclear
                return QueryType.SINGLE_STEP

        except Exception as e:
            print(f"Error in query routing: {e}")
            return QueryType.SINGLE_STEP

    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose a compound query into independent sub-queries.

        Args:
            query: The compound question

        Returns:
            List[str]: List of independent sub-questions
        """
        try:
            messages = self.decomposer_prompt.format_messages(question=query)
            response = self.llm.invoke(messages)

            # Parse JSON response
            import json

            sub_questions = json.loads(response.content.strip())

            if isinstance(sub_questions, list):
                return sub_questions
            else:
                return [query]  # Fallback to original query

        except Exception as e:
            print(f"Error in query decomposition: {e}")
            return [query]  # Fallback to original query

    def refine_seed_question(
        self, original_question: str, previous_steps: List[ReasoningStep] = None
    ) -> Union[str, bool]:
        """
        Refine a complex question into the next seed question.

        Args:
            original_question: The original complex question
            previous_steps: List of completed reasoning steps

        Returns:
            Union[str, bool]: Next seed question or True if sufficient
        """
        try:
            # Format previous steps
            steps_text = ""
            if previous_steps:
                for i, step in enumerate(previous_steps, 1):
                    steps_text += f"Step {i}: {step.question}\n"
                    steps_text += f"Answer: {step.intermediate_answer}\n\n"

            messages = self.refiner_prompt.format_messages(
                original_question=original_question,
                previous_steps=steps_text or "No previous steps",
            )
            response = self.llm.invoke(messages)
            result = response.content.strip()

            if "SUFFICIENT" in result.upper():
                return True
            else:
                return result

        except Exception as e:
            print(f"Error in seed question refinement: {e}")
            return original_question

    def discriminate_relevance(self, query: str, document: str) -> bool:
        """
        Determine if a document is relevant to answering a query.

        Args:
            query: The question
            document: The document content

        Returns:
            bool: True if relevant, False otherwise
        """
        try:
            messages = self.relevance_prompt.format_messages(
                question=query, document=document[:2000]  # Limit document length
            )
            response = self.llm.invoke(messages)
            result = response.content.strip().upper()

            return result == "RELEVANT"

        except Exception as e:
            print(f"Error in relevance discrimination: {e}")
            return True  # Default to relevant if error

    def should_end_reasoning(
        self, original_question: str, reasoning_steps: List[ReasoningStep]
    ) -> bool:
        """
        Determine if sufficient information has been gathered to answer the question.

        Args:
            original_question: The original question
            reasoning_steps: List of completed reasoning steps

        Returns:
            bool: True if sufficient, False if more reasoning needed
        """
        try:
            # Format reasoning steps
            steps_text = ""
            for i, step in enumerate(reasoning_steps, 1):
                steps_text += f"Step {i}: {step.question}\n"
                steps_text += f"Answer: {step.intermediate_answer}\n"
                steps_text += f"Confidence: {step.confidence}\n\n"

            messages = self.ending_prompt.format_messages(
                original_question=original_question,
                reasoning_steps=steps_text or "No steps completed",
            )
            response = self.llm.invoke(messages)
            result = response.content.strip().upper()

            return "SUFFICIENT" in result

        except Exception as e:
            print(f"Error in ending discrimination: {e}")
            return len(reasoning_steps) >= 3  # Default stopping criteria

    def filter_relevant_documents(
        self, query: str, documents: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Filter documents based on relevance to the query.

        Args:
            query: The question
            documents: List of retrieved documents

        Returns:
            List[RetrievalResult]: Filtered relevant documents
        """
        relevant_docs = []

        for doc in documents:
            if self.discriminate_relevance(query, doc.content):
                relevant_docs.append(doc)

        return relevant_docs

    async def process_compound_query_parallel(
        self, sub_queries: List[str], retriever, generator
    ) -> List[Dict[str, Any]]:
        """
        Process compound query sub-questions in parallel.

        Args:
            sub_queries: List of sub-questions
            retriever: Document retriever
            generator: Answer generator

        Returns:
            List[Dict]: Results for each sub-query
        """

        async def process_single_subquery(sub_query: str) -> Dict[str, Any]:
            """Process a single sub-query."""
            try:
                # Retrieve documents
                docs, is_noisy, noise_score = retriever.retrieve_with_noise_detection(
                    sub_query, k=config.top_k_documents
                )

                # Filter relevant documents
                relevant_docs = self.filter_relevant_documents(sub_query, docs)

                # Generate answer
                query_obj = MultiHopQuery(original_question=sub_query)
                response = generator.generate_with_noise_detection(
                    query_obj, relevant_docs, is_noisy
                )

                return {
                    "sub_query": sub_query,
                    "answer": response.answer,
                    "confidence": response.confidence,
                    "documents": relevant_docs,
                    "success": True,
                }

            except Exception as e:
                return {
                    "sub_query": sub_query,
                    "answer": f"Error processing: {str(e)}",
                    "confidence": 0.0,
                    "documents": [],
                    "success": False,
                }

        # Process all sub-queries in parallel
        tasks = [process_single_subquery(sub_query) for sub_query in sub_queries]
        results = await asyncio.gather(*tasks)

        return results

    def combine_compound_answers(
        self, original_query: str, sub_results: List[Dict[str, Any]]
    ) -> str:
        """
        Combine answers from compound query sub-questions.

        Args:
            original_query: The original compound question
            sub_results: Results from sub-query processing

        Returns:
            str: Combined final answer
        """
        try:
            # Create prompt for combining answers
            combine_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are an expert at combining multiple answers into a coherent response.

Given the original question and the answers to its sub-questions, provide a comprehensive final answer that addresses all aspects of the original question.

Guidelines:
1. Synthesize information from all sub-answers
2. Maintain coherence and flow
3. Address all parts of the original question
4. Be concise but complete

Format your response as a single, well-structured answer.""",
                    ),
                    (
                        "human",
                        """Original Question: {original_query}

Sub-questions and Answers:
{sub_answers}

Final Combined Answer:""",
                    ),
                ]
            )

            # Format sub-answers
            sub_answers_text = ""
            for i, result in enumerate(sub_results, 1):
                sub_answers_text += f"{i}. {result['sub_query']}\n"
                sub_answers_text += f"   Answer: {result['answer']}\n\n"

            messages = combine_prompt.format_messages(
                original_query=original_query, sub_answers=sub_answers_text
            )
            response = self.llm.invoke(messages)

            return response.content.strip()

        except Exception as e:
            print(f"Error combining compound answers: {e}")
            # Fallback: simple concatenation
            answers = [result["answer"] for result in sub_results if result["success"]]
            return " ".join(answers)
