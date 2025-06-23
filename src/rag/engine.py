from typing import AsyncIterator, Dict, Optional

from ..config.settings import settings
from ..core.interfaces import LLMInterface, StreamingChunk, VectorStoreInterface
from ..utils.logging import logger
from .reflexion_engine import ReflexionRAGEngine


class RAGEngine:
    """
    Main RAG Engine that directly uses ReflexionRAGEngine

    This class provides a simplified interface to the ReflexionRAGEngine,
    handling all RAG operations including document ingestion, querying,
    and memory management.
    """

    def __init__(
        self,
        generation_llm: Optional[LLMInterface] = None,
        vector_store: Optional[VectorStoreInterface] = None,
        **kwargs,
    ):
        """
        Initialize the RAG Engine

        Args:
            generation_llm: Optional LLM for generation, uses default if None
            vector_store: Optional vector store, uses default if None
            **kwargs: Additional arguments passed to ReflexionRAGEngine
        """
        logger.info("Initializing RAG Engine")
        self.engine = ReflexionRAGEngine(
            generation_llm=generation_llm, vector_store=vector_store, **kwargs
        )

    async def ingest_documents(self, directory_path: str) -> int:
        """
        Ingest documents from directory

        Args:
            directory_path: Path to directory containing documents to ingest

        Returns:
            Number of documents successfully ingested
        """
        logger.info("Ingesting documents", directory=directory_path)
        return await self.engine.ingest_documents(directory_path)

    async def query_stream(
        self, question: str, k: Optional[int] = None
    ) -> AsyncIterator[StreamingChunk]:
        """
        Process query using reflexion architecture with streaming response

        Args:
            question: User query to process
            k: Optional override for number of documents to retrieve

        Yields:
            StreamingChunk objects containing response content
        """
        logger.info("Processing query", question=question)
        async for chunk in self.engine.query_with_reflexion_stream(question):
            yield chunk

    def get_engine_info(self) -> Dict:
        """
        Return engine configuration information

        Returns:
            Dictionary containing engine configuration and statistics
        """
        return {
            "engine_type": "ReflexionRAGEngine",
            "max_reflexion_cycles": settings.max_reflexion_cycles,
            "confidence_threshold": settings.confidence_threshold,
            "memory_cache_enabled": settings.enable_memory_cache,
            "memory_stats": self.engine.get_memory_stats(),
        }

    async def clear_memory_cache(self) -> None:
        """
        Clear memory cache

        Removes all entries from the memory cache
        """
        logger.info("Clearing RAG engine memory cache")
        await self.engine.clear_memory_cache()

    async def count_documents(self) -> int:
        """
        Count documents in the vector store

        Returns:
            Number of documents in the vector store
        """
        logger.info("Counting documents in vector store")
        return await self.engine.vector_store.count_documents()

    async def count_web_searches(self) -> int:
        """
        Count search results in the vector store

        Return:
            Number of search results in vector store
        """
        logger.info("Counting search results in vector store")
        return await self.engine.vector_store.count_web_searches()

    async def delete_all_documents(self, confirm_string: str = "CONFIRM") -> bool:
        """
        Delete all documents from the vector store

        Args:
            confirm_string: Confirmation string (must be "CONFIRM" in caps)

        Returns:
            True if documents were deleted successfully, False otherwise
        """
        logger.info("Deleting all documents from vector store")
        return await self.engine.vector_store.delete_all_documents(confirm_string)
