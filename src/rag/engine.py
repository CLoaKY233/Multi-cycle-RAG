from typing import List, Optional

from ..config.settings import settings
from ..core.interfaces import (
    Document,
    LLMInterface,
    QueryResult,
    VectorStoreInterface,
)
from ..data.loader import DocumentLoader
from ..data.processor import DocumentProcessor
from ..llm.github_llm import GitHubLLM
from ..vectorstore.chroma_store import ChromaVectorStore


class RAGEngine:
    """Modular RAG engine"""

    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        vector_store: Optional[VectorStoreInterface] = None,
    ):
        self.llm = llm or GitHubLLM()
        self.vector_store = vector_store or ChromaVectorStore()
        self.document_loader = DocumentLoader()
        self.document_processor = DocumentProcessor()

    async def ingest_documents(self, directory_path: str) -> int:
        """Ingest documents from directory"""
        # Load Documents
        documents = await self.document_loader.load_from_directory(
            directory_path
        )

        # Process Documents (chunking, etc.)
        processed_docs = await self.document_processor.process_documents(
            documents
        )

        # Add to vector store
        doc_ids = await self.vector_store.add_documents(processed_docs)
        return len(doc_ids)

    async def query(
        self, question: str, k: Optional[int] = None
    ) -> QueryResult:
        """Query the RAG System"""
        k = k or settings.retrieval_k

        # Retrieve relevant documents
        relevant_docs = await self.vector_store.similarity_search(question, k=k)

        # Prepare context
        context = self._prepare_context(relevant_docs)

        # Generate Response
        prompt = self._create_prompt(question, context)
        response = await self.llm.generate(prompt)

        return QueryResult(
            response=response,
            source_documents=relevant_docs,
            metadata={"num_sources": len(relevant_docs), "query": question},
        )

    def _prepare_context(self, documents: List[Document]) -> str:
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            similarity = doc.metadata.get("similarity_score", 0)
            context_parts.append(
                f"Source {i} (similarity:{similarity:.3f}) - {source}:\n{doc.content}\n"
            )
        return "\n".join(context_parts)

    def _create_prompt(self, question: str, context: str) -> str:
        """Create a RAG Prompt"""
        return f"""
        You are an AI assistant that answers questions based on provided context.
        Use the context below to answer the question accurately and comprehensively.

        Context:
        {context}

        Question:{question}

        Instructions:
        - Answer based brimarily on provided context
        - if the context doesn't contain enough information, say so
        - Cite sources when possible
        - Be concise but thorough

        Answer:
        """
