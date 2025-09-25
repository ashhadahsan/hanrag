"""
LangGraph integration for HANRAG system.
Implements multi-hop reasoning workflow using LangGraph state management.
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback

from .models import (
    MultiHopQuery,
    HANRAGResponse,
    RetrievalResult,
    ReasoningStep,
    NoiseDetectionResult,
)
from .hanrag import HANRAGSystem
from .config import config


class HANRAGState(TypedDict):
    """State for HANRAG LangGraph workflow."""

    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    current_hop: int
    max_hops: int
    reasoning_steps: List[ReasoningStep]
    retrieved_documents: List[RetrievalResult]
    intermediate_answers: List[str]
    confidence_scores: List[float]
    is_query_noisy: bool
    noise_score: float
    final_answer: Optional[str]
    overall_confidence: float
    processing_metadata: Dict[str, Any]


class HANRAGLangGraphWorkflow:
    """LangGraph workflow for HANRAG multi-hop reasoning."""

    def __init__(self, hanrag_system: HANRAGSystem):
        """Initialize the LangGraph workflow."""
        self.hanrag_system = hanrag_system
        self.llm = ChatOpenAI(model_name=config.default_model, temperature=0.1)
        self.graph = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow."""
        workflow = StateGraph(HANRAGState)

        # Add nodes
        workflow.add_node("initialize", self._initialize_state)
        workflow.add_node("detect_noise", self._detect_query_noise)
        workflow.add_node("retrieve_documents", self._retrieve_documents)
        workflow.add_node("generate_reasoning", self._generate_reasoning)
        workflow.add_node("evaluate_confidence", self._evaluate_confidence)
        workflow.add_node("generate_final_answer", self._generate_final_answer)
        workflow.add_node("format_response", self._format_response)

        # Add edges
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "detect_noise")
        workflow.add_edge("detect_noise", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "generate_reasoning")
        workflow.add_edge("generate_reasoning", "evaluate_confidence")

        # Conditional edge for multi-hop reasoning
        workflow.add_conditional_edges(
            "evaluate_confidence",
            self._should_continue_reasoning,
            {"continue": "retrieve_documents", "finalize": "generate_final_answer"},
        )

        workflow.add_edge("generate_final_answer", "format_response")
        workflow.add_edge("format_response", END)

        return workflow.compile()

    def _initialize_state(self, state: HANRAGState) -> HANRAGState:
        """Initialize the workflow state."""
        # Extract question from messages
        if state["messages"]:
            last_message = state["messages"][-1]
            if isinstance(last_message, HumanMessage):
                state["question"] = last_message.content
            else:
                state["question"] = state.get("question", "")
        else:
            state["question"] = state.get("question", "")

        # Initialize state variables
        state["current_hop"] = 0
        state["max_hops"] = config.max_hops
        state["reasoning_steps"] = []
        state["retrieved_documents"] = []
        state["intermediate_answers"] = []
        state["confidence_scores"] = []
        state["is_query_noisy"] = False
        state["noise_score"] = 0.0
        state["final_answer"] = None
        state["overall_confidence"] = 0.0
        state["processing_metadata"] = {"workflow_started": True, "total_hops": 0}

        return state

    def _detect_query_noise(self, state: HANRAGState) -> HANRAGState:
        """Detect noise in the query."""
        question = state["question"]

        # Use HANRAG's noise detection
        _, is_noisy, noise_score = (
            self.hanrag_system.retriever.retrieve_with_noise_detection(question, k=1)
        )

        state["is_query_noisy"] = is_noisy
        state["noise_score"] = noise_score

        # Add system message about noise detection
        noise_message = SystemMessage(
            content=f"Query noise detected: {is_noisy}, score: {noise_score:.3f}"
        )
        state["messages"].append(noise_message)

        return state

    def _retrieve_documents(self, state: HANRAGState) -> HANRAGState:
        """Retrieve documents for current reasoning step."""
        question = state["question"]
        current_hop = state["current_hop"]

        # Retrieve documents using HANRAG retriever
        if current_hop == 0:
            # Initial retrieval
            docs, _, _ = self.hanrag_system.retriever.retrieve_with_noise_detection(
                question, k=config.top_k_documents
            )
        else:
            # Follow-up retrieval based on previous reasoning
            if state["reasoning_steps"]:
                last_step = state["reasoning_steps"][-1]
                follow_up_docs = self.hanrag_system._retrieve_follow_up_documents(
                    last_step, state["retrieved_documents"]
                )
                docs = self.hanrag_system._merge_documents(
                    state["retrieved_documents"], follow_up_docs
                )
            else:
                docs = state["retrieved_documents"]

        state["retrieved_documents"] = docs
        state["current_hop"] += 1

        # Add retrieval message
        retrieval_message = SystemMessage(
            content=f"Retrieved {len(docs)} documents for hop {current_hop + 1}"
        )
        state["messages"].append(retrieval_message)

        return state

    def _generate_reasoning(self, state: HANRAGState) -> HANRAGState:
        """Generate reasoning step for current hop."""
        question = state["question"]
        documents = state["retrieved_documents"]
        current_hop = state["current_hop"]

        # Generate reasoning step using HANRAG generator
        reasoning_step = self.hanrag_system._generate_reasoning(
            question, documents, current_hop
        )

        state["reasoning_steps"].append(reasoning_step)
        state["intermediate_answers"].append(reasoning_step.intermediate_answer or "")
        state["confidence_scores"].append(reasoning_step.confidence)

        # Add reasoning message
        reasoning_message = AIMessage(
            content=f"Step {current_hop}: {reasoning_step.question}\n"
            f"Reasoning: {reasoning_step.reasoning}\n"
            f"Answer: {reasoning_step.intermediate_answer}\n"
            f"Confidence: {reasoning_step.confidence:.3f}"
        )
        state["messages"].append(reasoning_message)

        return state

    def _evaluate_confidence(self, state: HANRAGState) -> HANRAGState:
        """Evaluate confidence and decide whether to continue reasoning."""
        current_confidence = (
            state["confidence_scores"][-1] if state["confidence_scores"] else 0.0
        )
        current_hop = state["current_hop"]
        max_hops = state["max_hops"]

        # Update processing metadata
        state["processing_metadata"]["total_hops"] = current_hop

        # Add confidence evaluation message
        confidence_message = SystemMessage(
            content=f"Confidence evaluation: {current_confidence:.3f}, "
            f"Hop: {current_hop}/{max_hops}"
        )
        state["messages"].append(confidence_message)

        return state

    def _should_continue_reasoning(self, state: HANRAGState) -> str:
        """Decide whether to continue reasoning or finalize."""
        current_confidence = (
            state["confidence_scores"][-1] if state["confidence_scores"] else 0.0
        )
        current_hop = state["current_hop"]
        max_hops = state["max_hops"]

        # Continue if confidence is low and we haven't reached max hops
        if current_confidence < config.confidence_threshold and current_hop < max_hops:
            return "continue"
        else:
            return "finalize"

    def _generate_final_answer(self, state: HANRAGState) -> HANRAGState:
        """Generate the final answer."""
        question = state["question"]
        reasoning_steps = state["reasoning_steps"]
        documents = state["retrieved_documents"]

        # Create multi-hop query
        query = MultiHopQuery(
            original_question=question,
            reasoning_steps=reasoning_steps,
            noise_detected=state["is_query_noisy"],
            noise_level=state["noise_score"],
        )

        # Generate final answer using HANRAG generator
        response = self.hanrag_system.generator.generate_with_noise_detection(
            query, documents, state["is_query_noisy"]
        )

        state["final_answer"] = response.answer
        state["overall_confidence"] = response.confidence

        # Add final answer message
        final_message = AIMessage(
            content=f"Final Answer: {response.answer}\n"
            f"Overall Confidence: {response.confidence:.3f}"
        )
        state["messages"].append(final_message)

        return state

    def _format_response(self, state: HANRAGState) -> HANRAGState:
        """Format the final response."""
        # Create HANRAG response
        query = MultiHopQuery(
            original_question=state["question"],
            reasoning_steps=state["reasoning_steps"],
            final_answer=state["final_answer"],
            confidence=state["overall_confidence"],
            noise_detected=state["is_query_noisy"],
            noise_level=state["noise_score"],
        )

        response = HANRAGResponse(
            query=query,
            answer=state["final_answer"] or "No answer generated",
            confidence=state["overall_confidence"],
            reasoning_chain=state["reasoning_steps"],
            retrieved_documents=state["retrieved_documents"],
            processing_time=0.0,  # Would be calculated in real implementation
            metadata=state["processing_metadata"],
        )

        # Add response to state
        state["processing_metadata"]["hanrag_response"] = response

        return state

    def run_workflow(self, question: str) -> HANRAGResponse:
        """Run the complete workflow for a question with LangSmith tracing."""
        # Initialize state
        initial_state = HANRAGState(
            messages=[HumanMessage(content=question)],
            question=question,
            current_hop=0,
            max_hops=config.max_hops,
            reasoning_steps=[],
            retrieved_documents=[],
            intermediate_answers=[],
            confidence_scores=[],
            is_query_noisy=False,
            noise_score=0.0,
            final_answer=None,
            overall_confidence=0.0,
            processing_metadata={},
        )

        # Run the workflow with LangSmith tracing
        with get_openai_callback() as cb:
            final_state = self.graph.invoke(initial_state)

        # Extract and return the HANRAG response
        response = final_state["processing_metadata"]["hanrag_response"]

        # Add token usage information to metadata
        if hasattr(response, "metadata"):
            response.metadata.update(
                {
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost": cb.total_cost,
                }
            )

        return response


class ConversationalHANRAGWorkflow(HANRAGLangGraphWorkflow):
    """Conversational version of HANRAG workflow with memory."""

    def __init__(self, hanrag_system: HANRAGSystem):
        """Initialize conversational workflow."""
        super().__init__(hanrag_system)
        self.conversation_memory = []
        self.graph = self._create_conversational_workflow()

    def _create_conversational_workflow(self) -> StateGraph:
        """Create conversational workflow with memory."""
        workflow = StateGraph(HANRAGState)

        # Add nodes
        workflow.add_node("initialize", self._initialize_conversation)
        workflow.add_node("check_context", self._check_conversation_context)
        workflow.add_node("detect_noise", self._detect_query_noise)
        workflow.add_node("retrieve_documents", self._retrieve_documents)
        workflow.add_node("generate_reasoning", self._generate_reasoning)
        workflow.add_node("evaluate_confidence", self._evaluate_confidence)
        workflow.add_node("generate_final_answer", self._generate_final_answer)
        workflow.add_node("update_memory", self._update_conversation_memory)
        workflow.add_node("format_response", self._format_response)

        # Add edges
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "check_context")
        workflow.add_edge("check_context", "detect_noise")
        workflow.add_edge("detect_noise", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "generate_reasoning")
        workflow.add_edge("generate_reasoning", "evaluate_confidence")

        # Conditional edge for multi-hop reasoning
        workflow.add_conditional_edges(
            "evaluate_confidence",
            self._should_continue_reasoning,
            {"continue": "retrieve_documents", "finalize": "generate_final_answer"},
        )

        workflow.add_edge("generate_final_answer", "update_memory")
        workflow.add_edge("update_memory", "format_response")
        workflow.add_edge("format_response", END)

        return workflow.compile()

    def _initialize_conversation(self, state: HANRAGState) -> HANRAGState:
        """Initialize conversational state."""
        # Initialize basic state
        state = self._initialize_state(state)

        # Add conversation context
        state["processing_metadata"]["conversation_turn"] = (
            len(self.conversation_memory) + 1
        )
        state["processing_metadata"]["has_previous_context"] = (
            len(self.conversation_memory) > 0
        )

        return state

    def _check_conversation_context(self, state: HANRAGState) -> HANRAGState:
        """Check if current question relates to previous conversation."""
        if not self.conversation_memory:
            return state

        # Simple context checking - in practice, you'd use more sophisticated methods
        current_question = state["question"].lower()
        previous_questions = [
            turn["question"].lower() for turn in self.conversation_memory[-3:]
        ]

        # Check for follow-up indicators
        follow_up_indicators = [
            "what about",
            "how about",
            "also",
            "additionally",
            "furthermore",
        ]
        is_follow_up = any(
            indicator in current_question for indicator in follow_up_indicators
        )

        # Check for pronoun references
        pronoun_references = ["it", "this", "that", "they", "them", "these", "those"]
        has_pronouns = any(
            pronoun in current_question.split() for pronoun in pronoun_references
        )

        state["processing_metadata"]["is_follow_up"] = is_follow_up or has_pronouns
        state["processing_metadata"]["context_relevant"] = True

        return state

    def _update_conversation_memory(self, state: HANRAGState) -> HANRAGState:
        """Update conversation memory with current turn."""
        turn_data = {
            "question": state["question"],
            "answer": state["final_answer"],
            "confidence": state["overall_confidence"],
            "reasoning_steps": len(state["reasoning_steps"]),
            "timestamp": state["processing_metadata"].get("timestamp", "unknown"),
        }

        self.conversation_memory.append(turn_data)

        # Keep only last 10 turns to manage memory
        if len(self.conversation_memory) > 10:
            self.conversation_memory = self.conversation_memory[-10:]

        return state

    def run_conversation(self, question: str) -> HANRAGResponse:
        """Run conversational workflow."""
        # Initialize state
        initial_state = HANRAGState(
            messages=[HumanMessage(content=question)],
            question=question,
            current_hop=0,
            max_hops=config.max_hops,
            reasoning_steps=[],
            retrieved_documents=[],
            intermediate_answers=[],
            confidence_scores=[],
            is_query_noisy=False,
            noise_score=0.0,
            final_answer=None,
            overall_confidence=0.0,
            processing_metadata={},
        )

        # Run the workflow
        final_state = self.graph.invoke(initial_state)

        # Extract and return the HANRAG response
        return final_state["processing_metadata"]["hanrag_response"]

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.conversation_memory.copy()

    def clear_conversation_memory(self):
        """Clear conversation memory."""
        self.conversation_memory = []
