"""
Traditional RAG system implementation for comparison with HANRAG.
"""

import time
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.models import RetrievalResult, DocumentType


class TraditionalRAGSystem:
    """
    Traditional RAG system for comparison with HANRAG.
    Uses standard retrieval and generation without noise resistance or multi-hop reasoning.
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-ada-002",
    ):
        """Initialize the traditional RAG system."""
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        self.vectorstore = None
        self.qa_chain = None

        # Standard RAG prompt template
        self.prompt_template = PromptTemplate(
            template="""Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            Context:
            {context}

            Question: {question}
            Answer: """,
            input_variables=["context", "question"],
        )

    def add_knowledge_base(self, documents: List[Document]):
        """Add documents to the knowledge base."""
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)

        # Create vector store
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

        # Create modern LCEL-based QA chain
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.qa_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using traditional RAG.

        Args:
            question: The question to answer

        Returns:
            Dictionary with answer and metadata
        """
        if not self.qa_chain:
            raise ValueError(
                "Knowledge base not initialized. Call add_knowledge_base first."
            )

        start_time = time.time()

        # Get answer from modern LCEL chain
        answer = self.qa_chain.invoke(question)

        # Get retrieved documents
        retrieved_docs = self.vectorstore.similarity_search(question, k=5)

        # Convert to RetrievalResult format for consistency
        retrieval_results = []
        for i, doc in enumerate(retrieved_docs):
            retrieval_results.append(
                RetrievalResult(
                    document_id=f"doc_{i}",
                    content=doc.page_content,
                    score=1.0 - (i * 0.1),  # Simple scoring
                    metadata=doc.metadata,
                    document_type=DocumentType.TEXT,
                )
            )

        processing_time = time.time() - start_time

        return {
            "answer": answer,
            "confidence": 0.8,  # Fixed confidence for traditional RAG
            "retrieved_documents": retrieval_results,
            "processing_time": processing_time,
            "reasoning_chain": [],  # No multi-hop reasoning
            "noise_detected": False,  # No noise detection
            "noise_level": 0.0,
            "metadata": {
                "system_type": "traditional_rag",
                "retrieval_method": "semantic_search",
                "generation_method": "single_pass",
            },
        }

    def batch_answer_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Answer multiple questions in batch."""
        responses = []

        for question in questions:
            try:
                response = self.answer_question(question)
                responses.append(response)
            except Exception as e:
                # Create error response
                error_response = {
                    "answer": f"Error processing question: {str(e)}",
                    "confidence": 0.0,
                    "retrieved_documents": [],
                    "processing_time": 0.0,
                    "reasoning_chain": [],
                    "noise_detected": False,
                    "noise_level": 0.0,
                    "metadata": {"error": str(e)},
                }
                responses.append(error_response)

        return responses
