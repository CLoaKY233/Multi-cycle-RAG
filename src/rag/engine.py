from typing import AsyncIterator, Dict, Optional

from src.config.settings import settings
from src.core.interfaces import LLMInterface, StreamingChunk, VectorStoreInterface
from src.rag.reflexion_engine import ReflexionRAGEngine
from src.utils.logging import logger


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

    async def delete_all_web_searches(self, confirm_string: str = "CONFIRM") -> bool:
        """
        Delete all web search results from the vector store

        Args:
            confirm_string: Confirmation string (must be "CONFIRM" in caps)

        Returns:
            True if web searches were deleted successfully, False otherwise
        """
        logger.info("Deleting all web search results from vector store")
        return await self.engine.vector_store.delete_all_web_searches(confirm_string)

    async def clear_qa_cache(self) -> bool:
        """
        Delete all entries from the QA semantic cache table.

        Returns:
            True if cache was cleared successfully
        """
        logger.info("Clearing QA semantic cache")
        return await self.engine.vector_store.clear_qa_cache()

    # Runtime Configuration Methods

    def set_web_search_mode(self, mode: str) -> None:
        """Set web search mode: 'off', 'initial_only', or 'every_cycle'"""
        from src.config.settings import WebSearchMode

        mode_map = {
            "off": WebSearchMode.OFF,
            "initial_only": WebSearchMode.INITIAL_ONLY,
            "every_cycle": WebSearchMode.EVERY_CYCLE,
        }
        if mode.lower() not in mode_map:
            raise ValueError(
                f"Invalid mode: {mode}. Must be one of: {list(mode_map.keys())}"
            )
        self.engine.set_web_search_mode(mode_map[mode.lower()])

    def set_max_cycles(self, cycles: int) -> None:
        """Set max reflexion cycles (1-10)"""
        self.engine.set_max_cycles(cycles)

    def set_confidence_threshold(self, threshold: float) -> None:
        """Set confidence threshold (0.0-1.0)"""
        self.engine.set_confidence_threshold(threshold)

    def set_qa_cache_enabled(self, enabled: bool) -> None:
        """Enable or disable QA semantic cache"""
        self.engine.set_qa_cache_enabled(enabled)

    def set_qa_cache_threshold(self, threshold: float) -> None:
        """Set QA cache similarity threshold (0.0-1.0)"""
        self.engine.set_qa_cache_threshold(threshold)

    def get_runtime_config(self) -> Dict:
        """Get current runtime configuration"""
        return self.engine.get_runtime_config()

    def get_qa_cache_stats(self) -> Dict:
        """Get QA cache statistics"""
        return self.engine.get_qa_cache_stats()
