"""
LangChain integration for HANRAG system.
Provides document processing, text splitting, and LLM interaction utilities.
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    DirectoryLoader,
)
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.tracers import LangChainTracer
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

from .models import RetrievalResult, DocumentType
from .config import config


class DocumentProcessor:
    """Handles document loading and processing using LangChain."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize document processor."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def load_documents_from_file(self, file_path: str) -> List[Document]:
        """Load documents from a single file."""
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        documents = loader.load()
        return self._add_metadata(documents, file_path)

    def load_documents_from_directory(
        self, directory_path: str, glob_pattern: str = "**/*.txt"
    ) -> List[Document]:
        """Load documents from a directory."""
        loader = DirectoryLoader(
            directory_path,
            glob=glob_pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        documents = loader.load()
        return self._add_metadata(documents, directory_path)

    def load_documents_from_urls(self, urls: List[str]) -> List[Document]:
        """Load documents from URLs."""
        loader = WebBaseLoader(urls)
        documents = loader.load()
        return self._add_metadata(documents, "web")

    def load_documents_from_text(
        self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[Document]:
        """Load documents from raw text."""
        documents = []
        for i, text in enumerate(texts):
            if metadata and i < len(metadata):
                doc_metadata = metadata[i].copy()
                # Only add default source if not provided
                if "source" not in doc_metadata:
                    doc_metadata["source"] = f"text_{i}"
            else:
                doc_metadata = {"source": f"text_{i}", "type": "text"}
            documents.append(Document(page_content=text, metadata=doc_metadata))

        return documents

    def _add_metadata(self, documents: List[Document], source: str) -> List[Document]:
        """Add metadata to documents."""
        for i, doc in enumerate(documents):
            if not doc.metadata:
                doc.metadata = {}

            doc.metadata.update(
                {
                    "id": f"{source}_{i}",
                    "source": source,
                    "chunk_index": i,
                    "total_chunks": len(documents),
                }
            )

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        return self.text_splitter.split_documents(documents)

    def create_document_from_hanrag_paper(self) -> List[Document]:
        """Create a document from the HANRAG paper content."""
        # This would contain the actual paper content
        paper_content = """
        HANRAG: Heuristic Accurate Noise-resistant Retrieval-Augmented Generation
        for Multi-hop Question Answering
        
        Abstract:
        This paper presents HANRAG, a novel approach to multi-hop question answering
        that combines heuristic-based retrieval with noise-resistant generation.
        The system addresses key challenges in retrieval-augmented generation
        including noise in retrieved documents and the need for accurate multi-hop reasoning.
        
        Introduction:
        Multi-hop question answering requires reasoning across multiple documents
        to arrive at the correct answer. Traditional retrieval-augmented generation
        systems often struggle with noisy retrieved documents and lack robust
        mechanisms for multi-hop reasoning.
        
        Method:
        HANRAG introduces several key innovations:
        1. Heuristic-based document retrieval that improves relevance
        2. Noise-resistant generation that filters out irrelevant information
        3. Multi-hop reasoning framework that breaks down complex questions
        4. Confidence estimation for answer quality assessment
        
        Results:
        Experimental results show that HANRAG outperforms baseline methods
        on standard multi-hop QA benchmarks, with particular improvements
        in handling noisy retrieval scenarios.
        """

        return self.load_documents_from_text(
            [paper_content],
            [{"type": "scientific_paper", "title": "HANRAG Paper", "year": "2024"}],
        )


class LangChainRetriever:
    """LangChain-based retriever with advanced features."""

    def __init__(
        self, embedding_model: Optional[str] = None, vectorstore_type: str = "faiss"
    ):
        """Initialize LangChain retriever."""
        self.embedding_model = embedding_model or config.embedding_model
        self.vectorstore_type = vectorstore_type

        # Initialize embeddings
        if self.embedding_model.startswith("text-embedding"):
            self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)

        self.vectorstore = None
        self.retrieval_chain = None

    def create_vectorstore(
        self, documents: List[Document], vectorstore_type: Optional[str] = None
    ) -> None:
        """Create vector store from documents."""
        vs_type = vectorstore_type or self.vectorstore_type

        if vs_type == "faiss":
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        elif vs_type == "chroma":
            self.vectorstore = Chroma.from_documents(documents, self.embeddings)
        else:
            raise ValueError(f"Unsupported vectorstore type: {vs_type}")

    def create_retrieval_chain(self, llm) -> None:
        """Create a modern LCEL-based retrieval chain for question answering."""
        if not self.vectorstore:
            raise ValueError("Vector store not created. Call create_vectorstore first.")

        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": config.top_k_documents}
        )

        # Create prompt template
        prompt_template = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # Create modern LCEL-based retrieval chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.retrieval_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    def create_conversational_chain(self, llm) -> dict:
        """Create a modern conversational retrieval chain using create_history_aware_retriever."""
        if not self.vectorstore:
            raise ValueError("Vector store not created. Call create_vectorstore first.")

        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": config.top_k_documents}
        )

        # Create prompt for generating search queries based on conversation history
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                (
                    "user",
                    "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
                ),
            ]
        )

        # Create history-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # Create prompt for answering questions based on retrieved context
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the user's questions based on the below context:\n\n{context}",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
            ]
        )

        # Create document chain
        document_chain = create_stuff_documents_chain(llm, qa_prompt)

        # Create the conversational retrieval chain
        conversational_chain = create_retrieval_chain(
            history_aware_retriever, document_chain
        )

        return {"chain": conversational_chain, "retriever": history_aware_retriever}

    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """Perform similarity search with LangSmith tracing."""
        if not self.vectorstore:
            raise ValueError("Vector store not created.")

        k = k or config.top_k_documents

        # Use LangSmith tracing for similarity search
        with get_openai_callback() as cb:
            results = self.vectorstore.similarity_search(query, k=k)

        return results

    def similarity_search_with_score(self, query: str, k: int = None) -> List[tuple]:
        """Perform similarity search with scores and LangSmith tracing."""
        if not self.vectorstore:
            raise ValueError("Vector store not created.")

        k = k or config.top_k_documents

        # Use LangSmith tracing for similarity search with scores
        with get_openai_callback() as cb:
            results = self.vectorstore.similarity_search_with_score(query, k=k)

        return results

    def max_marginal_relevance_search(
        self, query: str, k: int = None, fetch_k: int = None
    ) -> List[Document]:
        """Perform MMR search for diverse results with LangSmith tracing."""
        if not self.vectorstore:
            raise ValueError("Vector store not created.")

        k = k or config.top_k_documents
        fetch_k = fetch_k or k * 2

        # Use LangSmith tracing for MMR search
        with get_openai_callback() as cb:
            results = self.vectorstore.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k
            )

        return results


class LangChainHANRAGIntegration:
    """Integration layer between HANRAG and LangChain components."""

    def __init__(self, hanrag_system):
        """Initialize integration with HANRAG system."""
        self.hanrag_system = hanrag_system
        self.document_processor = DocumentProcessor()
        self.langchain_retriever = LangChainRetriever()

    def setup_knowledge_base_from_files(self, file_paths: List[str]) -> None:
        """Setup knowledge base from files using LangChain."""
        all_documents = []

        for file_path in file_paths:
            documents = self.document_processor.load_documents_from_file(file_path)
            all_documents.extend(documents)

        # Split documents into chunks
        chunked_documents = self.document_processor.split_documents(all_documents)

        # Create vector store
        self.langchain_retriever.create_vectorstore(chunked_documents)

        # Add to HANRAG system
        self.hanrag_system.add_knowledge_base(chunked_documents)

    def setup_knowledge_base_from_directory(
        self, directory_path: str, glob_pattern: str = "**/*.txt"
    ) -> None:
        """Setup knowledge base from directory using LangChain."""
        documents = self.document_processor.load_documents_from_directory(
            directory_path, glob_pattern
        )

        # Split documents into chunks
        chunked_documents = self.document_processor.split_documents(documents)

        # Create vector store
        self.langchain_retriever.create_vectorstore(chunked_documents)

        # Add to HANRAG system
        self.hanrag_system.add_knowledge_base(chunked_documents)

    def setup_knowledge_base_from_urls(self, urls: List[str]) -> None:
        """Setup knowledge base from URLs using LangChain."""
        documents = self.document_processor.load_documents_from_urls(urls)

        # Split documents into chunks
        chunked_documents = self.document_processor.split_documents(documents)

        # Create vector store
        self.langchain_retriever.create_vectorstore(chunked_documents)

        # Add to HANRAG system
        self.hanrag_system.add_knowledge_base(chunked_documents)

    def create_hybrid_retrieval_chain(self, llm):
        """Create a hybrid retrieval chain combining HANRAG and LangChain using modern LCEL."""
        # Create LangChain retrieval chain
        langchain_result = self.langchain_retriever.create_conversational_chain(llm)
        langchain_chain = langchain_result["chain"]

        # Create hybrid chain that uses both systems
        class HybridRetrievalChain:
            def __init__(self, hanrag_system, langchain_chain):
                self.hanrag_system = hanrag_system
                self.langchain_chain = langchain_chain
                self.chat_history = []  # Simple chat history storage

            def __call__(self, question: str):
                # Get HANRAG response
                hanrag_response = self.hanrag_system.answer_question(question)

                # Get LangChain response using modern LCEL
                langchain_result = self.langchain_chain.invoke(
                    {"input": question, "chat_history": self.chat_history}
                )
                langchain_answer = langchain_result["answer"]

                # Update chat history
                self.chat_history.append({"role": "user", "content": question})
                self.chat_history.append(
                    {"role": "assistant", "content": langchain_answer}
                )

                # Combine responses (simplified approach)
                combined_answer = f"""
                HANRAG Answer: {hanrag_response.answer}
                Confidence: {hanrag_response.confidence:.3f}
                
                LangChain Answer: {langchain_answer}
                
                Reasoning Steps: {len(hanrag_response.reasoning_chain)} steps
                """

                return {
                    "answer": combined_answer,
                    "hanrag_response": hanrag_response,
                    "langchain_response": langchain_answer,
                }

        return HybridRetrievalChain(self.hanrag_system, langchain_chain)

    def compare_retrieval_methods(self, question: str) -> Dict[str, Any]:
        """Compare HANRAG and LangChain retrieval methods."""
        # HANRAG retrieval
        hanrag_response = self.hanrag_system.answer_question(question)

        # LangChain retrieval
        langchain_docs = self.langchain_retriever.similarity_search(question)

        return {
            "question": question,
            "hanrag": {
                "answer": hanrag_response.answer,
                "confidence": hanrag_response.confidence,
                "reasoning_steps": len(hanrag_response.reasoning_chain),
                "retrieved_docs": len(hanrag_response.retrieved_documents),
                "processing_time": hanrag_response.processing_time,
            },
            "langchain": {
                "retrieved_docs": len(langchain_docs),
                "documents": [
                    doc.page_content[:200] + "..." for doc in langchain_docs[:3]
                ],
            },
        }
