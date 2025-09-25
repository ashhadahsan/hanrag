"""
Retrieval component for HANRAG system.
Implements heuristic-based document retrieval with noise resistance.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .models import RetrievalResult, DocumentType, HeuristicRule
from .config import config


class HeuristicRetriever:
    """Heuristic-based document retriever with noise resistance."""

    def __init__(self, embedding_model: Optional[str] = None):
        """Initialize the heuristic retriever."""
        self.embedding_model = embedding_model or config.embedding_model
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self.vectorstore: Optional[FAISS] = None
        self.heuristic_rules: List[HeuristicRule] = []
        self._initialize_heuristic_rules()

    def _initialize_heuristic_rules(self):
        """Initialize heuristic rules for improved retrieval."""
        self.heuristic_rules = [
            HeuristicRule(
                rule_id="entity_boost",
                name="Entity Mention Boost",
                description="Boost documents that contain entity mentions from the query",
                condition="entity_overlap > 0",
                action="score *= 1.2",
                weight=1.2,
            ),
            HeuristicRule(
                rule_id="temporal_relevance",
                name="Temporal Relevance",
                description="Boost recent documents for time-sensitive queries",
                condition="query_has_temporal_marker",
                action="recent_docs_score *= 1.1",
                weight=1.1,
            ),
            HeuristicRule(
                rule_id="domain_specificity",
                name="Domain Specificity",
                description="Boost documents from relevant domains",
                condition="domain_match",
                action="score *= 1.15",
                weight=1.15,
            ),
            HeuristicRule(
                rule_id="length_penalty",
                name="Length Penalty",
                description="Penalize very short or very long documents",
                condition="doc_length < 100 or doc_length > 5000",
                action="score *= 0.9",
                weight=0.9,
            ),
        ]

    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store."""
        if not self.vectorstore:
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vectorstore.add_documents(documents)

    def retrieve_documents(
        self, query: str, k: int = None, apply_heuristics: bool = True
    ) -> List[RetrievalResult]:
        """Retrieve documents using heuristic-enhanced similarity search."""
        if not self.vectorstore:
            raise ValueError("No documents have been added to the retriever")

        k = k or config.top_k_documents

        # Basic similarity search
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k * 2)

        # Apply heuristic rules if enabled
        if apply_heuristics:
            docs_with_scores = self._apply_heuristic_rules(query, docs_with_scores)

        # Convert to RetrievalResult objects
        results = []
        for doc, score in docs_with_scores[:k]:
            # Convert similarity score to relevance score (higher is better)
            relevance_score = 1.0 / (1.0 + score)

            result = RetrievalResult(
                document_id=doc.metadata.get("id", f"doc_{len(results)}"),
                content=doc.page_content,
                score=relevance_score,
                metadata=doc.metadata,
                document_type=DocumentType(doc.metadata.get("type", "text")),
            )
            results.append(result)

        return results

    def _apply_heuristic_rules(
        self, query: str, docs_with_scores: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """Apply heuristic rules to improve retrieval quality."""
        enhanced_docs = []

        for doc, score in docs_with_scores:
            enhanced_score = score

            # Apply entity boost rule
            if self._has_entity_overlap(query, doc.page_content):
                enhanced_score *= 0.8  # Lower distance = better score

            # Apply temporal relevance rule
            if self._has_temporal_marker(query):
                if self._is_recent_document(doc.metadata):
                    enhanced_score *= 0.9

            # Apply domain specificity rule
            if self._domain_matches(query, doc.metadata):
                enhanced_score *= 0.85

            # Apply length penalty rule
            doc_length = len(doc.page_content)
            if doc_length < 100 or doc_length > 5000:
                enhanced_score *= 1.1  # Penalty for inappropriate length

            enhanced_docs.append((doc, enhanced_score))

        # Sort by enhanced scores
        enhanced_docs.sort(key=lambda x: x[1])
        return enhanced_docs

    def _has_entity_overlap(self, query: str, content: str) -> bool:
        """Check if query and content share entity mentions."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        # Simple entity overlap detection
        overlap = len(query_words.intersection(content_words))
        return overlap > 0

    def _has_temporal_marker(self, query: str) -> bool:
        """Check if query contains temporal markers."""
        temporal_markers = [
            "recent",
            "latest",
            "new",
            "current",
            "today",
            "yesterday",
            "this year",
            "last year",
            "2024",
            "2023",
            "now",
        ]
        query_lower = query.lower()
        return any(marker in query_lower for marker in temporal_markers)

    def _is_recent_document(self, metadata: Dict[str, Any]) -> bool:
        """Check if document is recent based on metadata."""
        # This is a simplified check - in practice, you'd parse dates
        return metadata.get("recent", False)

    def _domain_matches(self, query: str, metadata: Dict[str, Any]) -> bool:
        """Check if document domain matches query domain."""
        query_domain = self._extract_domain_from_query(query)
        doc_domain = metadata.get("domain", "")
        return query_domain in doc_domain or doc_domain in query_domain

    def _extract_domain_from_query(self, query: str) -> str:
        """Extract domain from query (simplified implementation)."""
        # This is a simplified domain extraction
        domain_keywords = {
            "science": ["research", "study", "experiment", "scientific"],
            "technology": ["software", "computer", "AI", "machine learning"],
            "medicine": ["health", "medical", "disease", "treatment"],
            "business": ["company", "market", "finance", "economy"],
        }

        query_lower = query.lower()
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return domain
        return "general"


class NoiseResistantRetriever(HeuristicRetriever):
    """Noise-resistant version of the heuristic retriever."""

    def __init__(self, embedding_model: Optional[str] = None):
        super().__init__(embedding_model)
        self.noise_threshold = config.noise_threshold

    def retrieve_with_noise_detection(
        self, query: str, k: int = None
    ) -> Tuple[List[RetrievalResult], bool, float]:
        """Retrieve documents with noise detection."""
        # First, get initial retrieval results
        results = self.retrieve_documents(query, k)

        # Detect noise in the query
        is_noisy, noise_score = self._detect_query_noise(query)

        # If query is noisy, apply noise-resistant strategies
        if is_noisy:
            results = self._apply_noise_resistance(query, results)

        return results, is_noisy, noise_score

    def _detect_query_noise(self, query: str) -> Tuple[bool, float]:
        """Detect if query contains noise."""
        noise_indicators = [
            # Ambiguous terms
            "thing",
            "stuff",
            "something",
            "anything",
            # Vague temporal references
            "sometime",
            "recently",
            "lately",
            # Uncertain language
            "maybe",
            "perhaps",
            "might",
            "could be",
            # Incomplete phrases
            "what about",
            "how about",
            "tell me about",
        ]

        query_lower = query.lower()
        noise_count = sum(
            1 for indicator in noise_indicators if indicator in query_lower
        )
        noise_score = noise_count / len(noise_indicators)

        is_noisy = noise_score > config.query_noise_threshold
        return is_noisy, noise_score

    def _apply_noise_resistance(
        self, query: str, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Apply noise resistance strategies to retrieval results."""
        # Strategy 1: Expand query with synonyms
        expanded_query = self._expand_query(query)
        expanded_results = self.retrieve_documents(expanded_query, len(results))

        # Strategy 2: Combine and re-rank results
        combined_results = self._combine_and_rerank(results, expanded_results)

        # Strategy 3: Filter out low-confidence results
        filtered_results = [
            result
            for result in combined_results
            if result.score > config.similarity_threshold
        ]

        return filtered_results[: config.top_k_documents]

    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms."""
        # This is a simplified expansion - in practice, you'd use a thesaurus or LLM
        expansions = {
            "good": "excellent, great, positive, beneficial",
            "bad": "poor, negative, harmful, detrimental",
            "big": "large, huge, massive, significant",
            "small": "tiny, little, minor, insignificant",
        }

        expanded_query = query
        for term, synonyms in expansions.items():
            if term in query.lower():
                expanded_query += f" {synonyms}"

        return expanded_query

    def _combine_and_rerank(
        self, results1: List[RetrievalResult], results2: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Combine and re-rank results from different queries."""
        # Create a dictionary to track unique documents
        unique_docs = {}

        # Add results from first query
        for result in results1:
            unique_docs[result.document_id] = result

        # Add or update results from second query
        for result in results2:
            if result.document_id in unique_docs:
                # Take the higher score
                if result.score > unique_docs[result.document_id].score:
                    unique_docs[result.document_id] = result
            else:
                unique_docs[result.document_id] = result

        # Sort by score and return
        combined_results = list(unique_docs.values())
        combined_results.sort(key=lambda x: x.score, reverse=True)

        return combined_results

    def _filter_noisy_documents(
        self, documents: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Filter out documents that contain noise indicators."""
        filtered_docs = []
        for doc in documents:
            # Filter out documents with noise indicators or very short content
            if (
                not self._contains_noise_indicators(doc.content)
                and len(doc.content.strip()) > 50
            ):
                filtered_docs.append(doc)
        return filtered_docs

    def _contains_noise_indicators(self, content: str) -> bool:
        """Check if content contains noise indicators."""
        noise_indicators = [
            "error",
            "failed",
            "exception",
            "warning",
            "undefined",
            "null",
            "none",
            "missing",
            "not found",
            "invalid",
            "lorem ipsum",
            "placeholder",
            "dummy",
            "test",
        ]

        content_lower = content.lower()
        return any(indicator in content_lower for indicator in noise_indicators)
