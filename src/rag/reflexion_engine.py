import asyncio
import time
from datetime import datetime
from typing import AsyncIterator, Dict, List, Optional

from prompts.manager import prompt_manager

from ..config.settings import WebSearchMode, settings
from ..core.interfaces import (
    Document,
    LLMInterface,
    ReflexionCycle,
    ReflexionDecision,
    ReflexionEvaluation,
    ReflexionEvaluatorInterface,
    ReflexionMemory,
    StreamingChunk,
    VectorStoreInterface,
    WebSearchInterface,
    WebSearchResult,
    WebSearchStatus,
)
from ..data.loader import DocumentLoader
from ..data.processor import DocumentProcessor
from ..llm.github_llm import GitHubLLM
from ..memory.cache import ReflexionMemoryCache, create_query_hash
from ..reflexion.evaluator import SmartReflexionEvaluator
from ..utils.logging import logger
from ..vectorstore.surrealdb_store import SurrealDBVectorStore
from ..websearch.google_search import GoogleWebSearch


class ReflexionRAGEngine:
    """Advanced RAG engine with dynamic reflexion loop architecture and web search integration"""

    def __init__(
        self,
        generation_llm: Optional[LLMInterface] = None,
        evaluation_llm: Optional[LLMInterface] = None,
        summary_llm: Optional[LLMInterface] = None,
        vector_store: Optional[VectorStoreInterface] = None,
        reflexion_evaluator: Optional[ReflexionEvaluatorInterface] = None,
        web_search: Optional[WebSearchInterface] = None,
    ):
        # Initialize different LLMs for different purposes
        self.generation_llm = generation_llm or GitHubLLM(
            model_override=settings.llm_model,
            temperature_override=settings.llm_temperature,
            max_tokens_override=settings.llm_max_tokens,
        )

        self.evaluation_llm = evaluation_llm or GitHubLLM(
            model_override=settings.evaluation_model,
            temperature_override=settings.evaluation_temperature,
            max_tokens_override=settings.evaluation_max_tokens,
        )

        self.summary_llm = summary_llm or GitHubLLM(
            model_override=settings.summary_model,
            temperature_override=settings.summary_temperature,
            max_tokens_override=settings.summary_max_tokens,
        )

        self.vector_store = vector_store or SurrealDBVectorStore()
        self.reflexion_evaluator = reflexion_evaluator or SmartReflexionEvaluator(
            self.evaluation_llm
        )

        # Web search integration
        self.web_search = web_search or GoogleWebSearch()

        # Memory cache for reflexion loops
        self.memory_cache = (
            ReflexionMemoryCache() if settings.enable_memory_cache else None
        )

        # Document processing
        self.document_loader = DocumentLoader()
        self.document_processor = DocumentProcessor()

    async def ingest_documents(self, directory_path: str) -> int:
        """Ingest documents from directory

        Args:
            directory_path: Path to directory containing documents

        Returns:
            Number of documents successfully ingested
        """
        documents = await self.document_loader.load_from_directory(directory_path)
        processed_docs = await self.document_processor.process_documents(documents)
        doc_ids = await self.vector_store.add_documents(processed_docs)
        return len(doc_ids)

    async def query_with_reflexion_stream(
        self, question: str
    ) -> AsyncIterator[StreamingChunk]:
        """Main reflexion loop query with streaming and web search integration"""

        logger.info(
            "Starting Reflexion Loop",
            query=question,
            web_mode=settings.web_search_mode.value,
        )

        # Check memory cache first
        query_hash = create_query_hash(question)
        cached_memory = None
        if self.memory_cache:
            try:
                cached_memory = self.memory_cache.get(query_hash)
                if cached_memory:
                    logger.info("Found cached reflexion result", query_hash=query_hash)
                    async for chunk in self._stream_cached_result(cached_memory):
                        yield chunk
                    return
            except Exception as e:
                logger.warning(
                    "Cache retrieval error (continuing without cache)",
                    error=str(e),
                )

        # Initialize reflexion memory
        reflexion_memory = ReflexionMemory(original_query=question)
        start_time = time.time()

        try:
            # Reflexion loop
            cycle_number = 1
            current_query = question

            while cycle_number <= settings.max_reflexion_cycles:
                logger.info("Starting reflexion cycle", cycle=cycle_number)
                logger.info("Processing query", query=current_query, cycle=cycle_number)

                cycle_start = time.time()
                web_search_results = []

                # Step 1: Determine if web search should be performed
                should_perform_web_search = self._should_perform_web_search(
                    cycle_number
                )

                # Step 2: Parallel retrieval from DB and optionally web
                try:
                    k = (
                        settings.initial_retrieval_k
                        if cycle_number == 1
                        else settings.reflexion_retrieval_k
                    )

                    if should_perform_web_search:
                        # Use combined search with token limits
                        logger.info(
                            "Performing combined DB and web search",
                            cycle=cycle_number,
                        )

                        # Use the combined search method with proper limits
                        k_docs = min(k, 2)  # Limit DB docs
                        k_web = min(
                            settings.web_search_retrieval_k, 2
                        )  # Limit web docs

                        try:
                            retrieved_docs = (
                                await self.vector_store.similarity_search_combined(
                                    current_query, k_docs=k_docs, k_web=k_web
                                )
                            )

                            # Separate for reporting
                            db_docs = [
                                d
                                for d in retrieved_docs
                                if d.metadata.get("source_type") == "document"
                            ]
                            web_docs = [
                                d
                                for d in retrieved_docs
                                if d.metadata.get("source_type") == "web_search"
                            ]

                            # For web search results, we still need to perform the search for new content
                            web_search_results = await self._perform_web_search(
                                current_query
                            )

                            logger.info(
                                "Combined retrieval completed",
                                db_docs=len(db_docs),
                                web_docs=len(web_docs),
                                new_web_results=len(web_search_results),
                                cycle=cycle_number,
                            )

                        except Exception as e:
                            logger.error("Combined retrieval error", error=str(e))
                            retrieved_docs = []
                            web_search_results = []

                        # Store successful web search results in database
                        if web_search_results:
                            successful_results = [
                                r
                                for r in web_search_results
                                if r.status == WebSearchStatus.SUCCESS
                            ]
                            if successful_results:
                                try:
                                    await self.vector_store.add_web_search_results(
                                        successful_results
                                    )
                                    logger.info(
                                        f"Stored {len(successful_results)} web search results in database"
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"Failed to store web search results: {e}"
                                    )

                    else:
                        # DB search only with token limits
                        retrieved_docs = await self.vector_store.similarity_search(
                            current_query, k=min(k, 3)
                        )
                        logger.info(
                            "Retrieved documents from DB only",
                            count=len(retrieved_docs),
                        )

                except Exception as e:
                    logger.error("Retrieval error", error=str(e))
                    retrieved_docs = []
                    web_search_results = []
                    logger.warning("Proceeding with empty document set")

                # Step 3: Combine context from both sources with token limits
                combined_context = self._prepare_combined_context(
                    retrieved_docs, web_search_results
                )
                generation_prompt = self._create_generation_prompt(
                    current_query, combined_context, cycle_number
                )

                # Check prompt token count before generation
                prompt_tokens = self._estimate_tokens(generation_prompt)
                if prompt_tokens > 6000:  # Leave room for response
                    logger.warning(
                        f"Prompt too long ({prompt_tokens} tokens), truncating context"
                    )
                    # Truncate context and regenerate prompt
                    truncated_context = self._truncate_context(
                        combined_context, max_tokens=4000
                    )
                    generation_prompt = self._create_generation_prompt(
                        current_query, truncated_context, cycle_number
                    )

                # Step 4: Generate partial answer
                partial_answer_chunks = []
                try:
                    async for chunk in self.generation_llm.generate_stream(
                        generation_prompt
                    ):
                        partial_answer_chunks.append(chunk.content)
                        # Stream intermediate answers
                        if chunk.content:
                            yield StreamingChunk(
                                content=chunk.content,
                                metadata={
                                    "cycle_number": cycle_number,
                                    "is_partial": True,
                                    "reflexion_mode": True,
                                    "web_search_enabled": should_perform_web_search,
                                    "web_results_count": len(web_search_results),
                                },
                            )
                    yield StreamingChunk(
                        content="\n",  # Add newline buffer
                        metadata={
                            "cycle_number": cycle_number,
                            "is_spacing": True,
                            "reflexion_mode": True,
                        },
                    )
                except Exception as e:
                    logger.error("Generation error", error=str(e), cycle=cycle_number)
                    partial_answer_chunks = ["Error generating response."]
                    yield StreamingChunk(
                        content="⚠️ Error during response generation. Attempting recovery...",
                        metadata={
                            "cycle_number": cycle_number,
                            "is_partial": True,
                            "is_error": True,
                            "reflexion_mode": True,
                        },
                    )

                partial_answer = "".join(partial_answer_chunks)

                # Check if response appears truncated and handle continuation
                if self._is_likely_truncated(partial_answer):
                    logger.warning(
                        "Response appears truncated, attempting continuation",
                        cycle=cycle_number,
                    )
                    continuation_prompt = f"""Continue this response where it left off. Maintain the same tone and style:

                    PREVIOUS RESPONSE:
                    {partial_answer}

                    CONTINUE FROM WHERE IT STOPPED:"""

                    continuation_chunks = []
                    try:
                        async for chunk in self.generation_llm.generate_stream(
                            continuation_prompt,
                            max_tokens=min(settings.llm_max_tokens, 1500),
                        ):
                            continuation_chunks.append(chunk.content)
                            if chunk.content:
                                yield StreamingChunk(
                                    content=chunk.content,
                                    metadata={
                                        "cycle_number": cycle_number,
                                        "is_partial": True,
                                        "is_continuation": True,
                                        "reflexion_mode": True,
                                    },
                                )

                        continuation = "".join(continuation_chunks)
                        partial_answer = partial_answer + " " + continuation
                    except Exception as e:
                        logger.error(f"Continuation failed: {e}")

                logger.info(
                    "Generated answer",
                    chars=len(partial_answer),
                    cycle=cycle_number,
                )

                # Step 5: Self-evaluation
                logger.info("Evaluating response quality", cycle=cycle_number)
                try:
                    evaluation = await self.reflexion_evaluator.evaluate_response(
                        question,
                        partial_answer,
                        retrieved_docs,
                        cycle_number,
                    )
                    logger.info(
                        "Evaluation complete",
                        confidence=f"{evaluation.confidence_score:.2f}",
                        decision=evaluation.decision.name,
                        cycle=cycle_number,
                    )
                    logger.debug("Evaluation reasoning", reasoning=evaluation.reasoning)
                except Exception as e:
                    logger.error("Evaluation error", error=str(e))
                    evaluation = ReflexionEvaluation(
                        confidence_score=0.5,
                        decision=ReflexionDecision.CONTINUE,
                        reasoning=f"Evaluation failed with error: {e}",
                        follow_up_queries=[],
                        covered_aspects=[],
                        missing_aspects=["evaluation_error"],
                        uncertainty_phrases=[],
                        metadata={"error": str(e)},
                    )

                # Create reflexion cycle
                try:
                    cycle = ReflexionCycle(
                        cycle_number=cycle_number,
                        query=current_query,
                        retrieved_docs=retrieved_docs,
                        web_search_results=web_search_results,
                        partial_answer=partial_answer,
                        evaluation=evaluation,
                        timestamp=datetime.now(),
                        processing_time=time.time() - cycle_start,
                        web_search_enabled=should_perform_web_search,
                    )
                    reflexion_memory.add_cycle(cycle)
                except Exception as e:
                    logger.error(
                        "Error adding cycle to memory",
                        error=str(e),
                        cycle=cycle_number,
                    )

                # Step 6: Decision tree
                if evaluation.decision == ReflexionDecision.INSUFFICIENT_DATA:
                    logger.warning(
                        "Insufficient data in knowledge base",
                        cycle=cycle_number,
                    )
                    reflexion_memory.final_answer = (
                        self._create_insufficient_data_response(
                            question, reflexion_memory.get_all_partial_answers()
                        )
                    )
                    break

                elif evaluation.confidence_score >= settings.confidence_threshold:
                    logger.info(
                        "Confidence threshold reached",
                        confidence=f"{evaluation.confidence_score:.2f}",
                        threshold=settings.confidence_threshold,
                        cycle=cycle_number,
                    )
                    reflexion_memory.final_answer = partial_answer
                    break

                elif (
                    evaluation.decision == ReflexionDecision.COMPLETE
                    and evaluation.confidence_score
                    >= settings.confidence_threshold * 0.9
                ):
                    logger.info(
                        "Response is complete with sufficient confidence",
                        confidence=f"{evaluation.confidence_score:.2f}",
                        cycle=cycle_number,
                    )
                    reflexion_memory.final_answer = partial_answer
                    break

                elif cycle_number >= settings.max_reflexion_cycles:
                    logger.info(
                        "Max cycles reached, synthesizing final answer",
                        cycles=cycle_number,
                    )
                    break

                else:
                    # Continue with follow-up queries
                    try:
                        if evaluation.follow_up_queries:
                            current_query = evaluation.follow_up_queries[0]
                            logger.info(
                                "Following up with generated query",
                                query=current_query,
                                cycle=cycle_number,
                            )
                        else:
                            try:
                                follow_ups = await self.reflexion_evaluator.generate_follow_up_queries(
                                    question,
                                    partial_answer,
                                    evaluation.missing_aspects,
                                )
                                if follow_ups:
                                    current_query = follow_ups[0]
                                    logger.info(
                                        "Generated follow-up query",
                                        query=current_query,
                                        cycle=cycle_number,
                                    )
                                else:
                                    logger.warning(
                                        "No follow-up queries generated, stopping",
                                        cycle=cycle_number,
                                    )
                                    reflexion_memory.final_answer = partial_answer
                                    break
                            except Exception as e:
                                logger.error(
                                    "Error generating follow-up queries",
                                    error=str(e),
                                    cycle=cycle_number,
                                )
                                reflexion_memory.final_answer = partial_answer
                                break
                    except Exception as e:
                        logger.error(
                            "Error in follow-up query handling",
                            error=str(e),
                            cycle=cycle_number,
                        )
                        reflexion_memory.final_answer = partial_answer
                        break

                cycle_number += 1
                await asyncio.sleep(0.1)  # Brief pause between cycles

            # Final synthesis if needed
            if not reflexion_memory.final_answer and len(reflexion_memory.cycles) > 1:
                logger.info("Synthesizing final comprehensive answer")
                try:
                    reflexion_memory.final_answer = await self._synthesize_final_answer(
                        question, reflexion_memory
                    )
                except Exception as e:
                    logger.error("Error synthesizing final answer", error=str(e))
                    best_cycle = max(
                        reflexion_memory.cycles,
                        key=lambda c: c.evaluation.confidence_score,
                    )
                    reflexion_memory.final_answer = (
                        best_cycle.partial_answer or "Error synthesizing final answer."
                    )

            # Stream final answer
            final_answer = (
                reflexion_memory.final_answer or "Unable to generate a complete answer."
            )

            # Check if final answer appears truncated
            if self._is_likely_truncated(final_answer):
                logger.warning(
                    "Final answer appears truncated, attempting to complete it"
                )
                completion_prompt = f"""Complete this response that was cut off. Maintain the same tone and style:

                ORIGINAL QUESTION: {question}

                CURRENT RESPONSE:
                {final_answer}

                COMPLETE THE RESPONSE:"""

                completion_chunks = []
                try:
                    async for chunk in self.generation_llm.generate_stream(
                        completion_prompt,
                        max_tokens=min(settings.summary_max_tokens, 2000),
                    ):
                        completion_chunks.append(chunk.content)

                    completion = "".join(completion_chunks)
                    final_answer = final_answer + " " + completion
                except Exception as e:
                    logger.error(f"Final completion failed: {e}")

            # Add processing metadata
            reflexion_memory.total_processing_time = time.time() - start_time

            # Cache the result
            if self.memory_cache:
                try:
                    self.memory_cache.put(query_hash, reflexion_memory)
                except Exception as e:
                    logger.error("Error caching result", error=str(e))

            # Stream final answer with metadata
            try:
                yield StreamingChunk(
                    content=f"\n\n## 🎯 Final Comprehensive Answer\n\n{final_answer}",
                    is_complete=True,
                    metadata={
                        "reflexion_complete": True,
                        "total_cycles": len(reflexion_memory.cycles),
                        "total_processing_time": reflexion_memory.total_processing_time,
                        "total_documents": reflexion_memory.total_documents_retrieved,
                        "total_web_results": reflexion_memory.total_web_results_retrieved,
                        "final_confidence": reflexion_memory.cycles[
                            -1
                        ].evaluation.confidence_score
                        if reflexion_memory.cycles
                        else 0.0,
                        "memory_cached": self.memory_cache is not None,
                        "web_search_used": any(
                            cycle.web_search_enabled
                            for cycle in reflexion_memory.cycles
                        ),
                    },
                )
            except Exception as e:
                logger.error("Error streaming final result", error=str(e))
                yield StreamingChunk(
                    content=f"\n\n## Final Answer\n\n{final_answer}",
                    is_complete=True,
                    metadata={"reflexion_error": str(e)},
                )

        except Exception as e:
            logger.error("Reflexion loop failed", error=str(e), query=question)
            yield StreamingChunk(
                content=f"\n\n⚠️ **Error in reflexion process:** {e}\n\nFalling back to simple RAG...\n\n",
                metadata={"error": str(e), "is_error": True},
            )

            # Fallback to simple query
            try:
                logger.info("Attempting fallback simple query")
                async for chunk in self.simple_query_stream(question):
                    yield chunk
            except Exception as fallback_error:
                logger.critical(
                    "Both reflexion and fallback methods failed",
                    original_error=str(e),
                    fallback_error=str(fallback_error),
                )
                yield StreamingChunk(
                    content=f"\n\n❌ **Critical error:** Both reflexion and fallback methods failed. Error: {fallback_error}",
                    is_complete=True,
                    metadata={"critical_error": str(fallback_error)},
                )

    def _should_perform_web_search(self, cycle_number: int) -> bool:
        """Determine if web search should be performed for this cycle"""
        if settings.web_search_mode == WebSearchMode.OFF:
            return False
        elif settings.web_search_mode == WebSearchMode.INITIAL_ONLY:
            return cycle_number == 1
        elif settings.web_search_mode == WebSearchMode.EVERY_CYCLE:
            return True
        return False

    async def _perform_web_search(self, query: str) -> List[WebSearchResult]:
        """Perform web search and return results"""
        if not await self.web_search.is_available():
            logger.warning("Web search not available")
            return []

        try:
            web_results = await self.web_search.search_and_extract(
                query,
                num_results=min(settings.web_search_results_count, 3),  # Limit results
            )

            # Filter out failed results if needed
            successful_results = [
                r for r in web_results if r.status != WebSearchStatus.ERROR
            ]
            logger.info(
                f"Web search completed: {len(successful_results)}/{len(web_results)} successful"
            )

            return web_results
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)"""
        return len(text) // 4

    def _truncate_context(self, context: str, max_tokens: int = 4000) -> str:
        """Truncate context to fit within token limits"""
        max_chars = max_tokens * 4
        if len(context) <= max_chars:
            return context

        # Try to truncate at document boundary
        truncated = context[:max_chars]
        last_separator = truncated.rfind("-" * 80)  # Document separator
        if last_separator > max_chars * 0.7:  # If we can find a good break point
            truncated = truncated[:last_separator]

        return truncated + "\n\n[Context truncated due to token limits]"

    def _prepare_combined_context(
        self, db_docs: List[Document], web_results: List[WebSearchResult]
    ) -> str:
        """Prepare combined context with token limits"""
        context_parts = []
        total_tokens = 0
        max_context_tokens = 4000  # Leave room for question and instructions

        # Add database documents first (prioritize existing knowledge)
        if db_docs:
            context_parts.append("## Knowledge Base Documents\n")
            for i, doc in enumerate(db_docs, 1):
                doc_content = self._format_document_for_context(doc, i)
                doc_tokens = self._estimate_tokens(doc_content)

                if total_tokens + doc_tokens > max_context_tokens:
                    context_parts.append(
                        f"\n[Additional {len(db_docs) - i + 1} documents truncated due to length limits]"
                    )
                    break

                context_parts.append(doc_content)
                total_tokens += doc_tokens

        # Add web search results if we have token budget
        if web_results and total_tokens < max_context_tokens:
            successful_web_results = [
                r for r in web_results if r.status == WebSearchStatus.SUCCESS
            ]
            if successful_web_results:
                context_parts.append("\n\n## Web Search Results\n")
                for i, result in enumerate(successful_web_results, 1):
                    web_content = self._format_web_result_for_context(result, i)
                    web_tokens = self._estimate_tokens(web_content)

                    if total_tokens + web_tokens > max_context_tokens:
                        context_parts.append(
                            f"\n[Additional {len(successful_web_results) - i + 1} web results truncated due to length limits]"
                        )
                        break

                    context_parts.append(web_content)
                    total_tokens += web_tokens

        final_context = (
            "\n\n".join(context_parts)
            if context_parts
            else "No relevant documents found."
        )
        logger.info(
            f"Prepared context with estimated {self._estimate_tokens(final_context)} tokens"
        )

        return final_context

    def _format_document_for_context(self, doc: Document, index: int) -> str:
        """Format document for context with length limits"""
        metadata = doc.metadata
        source = metadata.get("source", "Unknown")
        file_name = metadata.get("file_name", "Unknown")
        file_type = metadata.get("file_type", "Unknown")
        creation_date = metadata.get("creation_date", "Unknown")
        similarity = metadata.get("similarity_score", 0)

        # Truncate content if needed
        content = doc.content
        if len(content) > 1500:  # Limit per document
            content = content[:1500] + "\n[Content truncated]"

        doc_header = f"""Document {index} [Similarity: {similarity:.3f}]
        ├─ File: {file_name} ({file_type})
        ├─ Created: {creation_date}
        ├─ Location: {source}
        """
        return f"{doc_header}\n\nContent:\n{content}\n{'-' * 80}"

    def _format_web_result_for_context(
        self, result: WebSearchResult, index: int
    ) -> str:
        """Format web result for context with length limits"""
        # Truncate content if needed
        content = result.content
        if len(content) > 1000:  # Smaller limit for web content
            content = content[:1000] + "\n[Content truncated]"

        web_header = f"""Web Result {index} [Rank: {result.rank}]
        ├─ Title: {result.title}
        ├─ URL: {result.url}
        ├─ Word Count: {result.word_count}
        ├─ Extraction: {result.extraction_strategy}
        """
        return f"{web_header}\n\nContent:\n{content}\n{'-' * 80}"

    def _is_likely_truncated(self, response: str) -> bool:
        """Check if response appears to be truncated"""
        if not response or len(response.strip()) < 50:
            return False

        if not any(response.strip().endswith(end) for end in [".", "!", "?", ":", ";"]):
            return True

        truncation_indicators = [
            "...",
            "[truncated]",
            "[cut off]",
            "due to length",
            "character limit",
            "token limit",
        ]

        if any(indicator in response.lower() for indicator in truncation_indicators):
            return True

        return False

    async def simple_query_stream(self, question: str) -> AsyncIterator[StreamingChunk]:
        """Simple RAG query without reflexion (fallback)"""
        logger.info("Using simple RAG mode", query=question)

        retrieved_docs = await self.vector_store.similarity_search(
            question, k=min(settings.initial_retrieval_k, 3)
        )
        logger.debug("Retrieved documents for simple query", count=len(retrieved_docs))
        context = self._prepare_context(retrieved_docs)
        prompt = self._create_simple_prompt(question, context)

        # Check token limits for simple query too
        prompt_tokens = self._estimate_tokens(prompt)
        if prompt_tokens > 6000:
            logger.warning(
                f"Simple prompt too long ({prompt_tokens} tokens), truncating"
            )
            context = self._truncate_context(context, max_tokens=3000)
            prompt = self._create_simple_prompt(question, context)

        async for chunk in self.generation_llm.generate_stream(prompt):
            if chunk.content:
                yield StreamingChunk(
                    content=chunk.content,
                    metadata={
                        "simple_mode": True,
                        "num_sources": len(retrieved_docs),
                    },
                )

    async def _stream_cached_result(
        self, memory: ReflexionMemory
    ) -> AsyncIterator[StreamingChunk]:
        """Stream cached reflexion result"""
        for i, cycle in enumerate(memory.cycles, 1):
            yield StreamingChunk(
                content=f"\n## 🔄 Cycle {i} (Cached)\n\n{cycle.partial_answer}\n",
                metadata={
                    "cycle_number": i,
                    "is_cached": True,
                    "confidence": cycle.evaluation.confidence_score,
                    "web_search_enabled": cycle.web_search_enabled,
                    "web_results_count": len(cycle.web_search_results),
                },
            )
            await asyncio.sleep(0.1)

        yield StreamingChunk(
            content=f"\n## 🎯 Final Answer (Cached)\n\n{memory.final_answer}",
            is_complete=True,
            metadata={
                "cached_result": True,
                "total_cycles": len(memory.cycles),
                "total_processing_time": memory.total_processing_time,
                "total_web_results": memory.total_web_results_retrieved,
            },
        )

    async def _synthesize_final_answer(
        self, question: str, memory: ReflexionMemory
    ) -> str:
        """Synthesize final answer from all reflexion cycles"""
        partial_answers = memory.get_all_partial_answers()
        all_docs = memory.get_all_retrieved_docs()
        all_web_results = memory.get_all_web_results()

        synthesis_prompt = self._create_synthesis_prompt(
            question, partial_answers, all_docs, memory.cycles, all_web_results
        )

        # Check token limits for synthesis
        prompt_tokens = self._estimate_tokens(synthesis_prompt)
        if prompt_tokens > 6000:
            logger.warning(
                f"Synthesis prompt too long ({prompt_tokens} tokens), using best cycle answer"
            )
            best_cycle = max(memory.cycles, key=lambda c: c.evaluation.confidence_score)
            return best_cycle.partial_answer

        answer_chunks = []
        async for chunk in self.summary_llm.generate_stream(synthesis_prompt):
            answer_chunks.append(chunk.content)

        return "".join(answer_chunks)

    def _create_generation_prompt(
        self, query: str, context: str, cycle_number: int
    ) -> str:
        """Create enhanced prompt for answer generation using prompt manager"""
        if cycle_number == 1:
            prompt_name = "initial_generation"
        else:
            prompt_name = "reflexion_generation"

        try:
            return prompt_manager.render_prompt(
                prompt_name,
                query=query,
                context=context,
                cycle_number=cycle_number,
            )
        except Exception as e:
            logger.error(f"Failed to render prompt {prompt_name}: {e}")
            return self._create_simple_prompt(query, context)

    def _create_synthesis_prompt(
        self,
        question: str,
        partial_answers: List[str],
        all_docs: List[Document],
        cycles: List[ReflexionCycle],
        all_web_results: List[WebSearchResult],
    ) -> str:
        """Create synthesis prompt using prompt manager"""
        answers_text = "\n\n".join(
            [
                f"Cycle {i + 1} Answer:\n{answer}"
                for i, answer in enumerate(partial_answers)
            ]
        )

        # Create deduplicated document reference
        unique_sources = {}
        for i, doc in enumerate(all_docs, 1):
            metadata = doc.metadata
            file_name = metadata.get("file_name", "Unknown")
            creation_date = metadata.get("creation_date", "Unknown")
            source = metadata.get("source", "Unknown")

            if file_name not in unique_sources:
                unique_sources[file_name] = {
                    "creation_date": creation_date,
                    "source": source,
                    "doc_numbers": [i],
                }
            else:
                unique_sources[file_name]["doc_numbers"].append(i)

        doc_references = []
        for file_name, info in unique_sources.items():
            doc_nums = ", ".join([f"Doc {num}" for num in info["doc_numbers"]])
            doc_references.append(
                f"Source: {file_name} (Created: {info['creation_date']}) - {info['source']} [Appears as: {doc_nums}]"
            )

        # Add web search sources
        if all_web_results:
            doc_references.append("\nWeb Search Sources:")
            for i, result in enumerate(all_web_results, 1):
                doc_references.append(f"Web {i}: {result.title} - {result.url}")

        references_text = "\n".join(doc_references)

        # Get evaluation insights
        evaluation_insights = []
        for cycle in cycles:
            eval_info = f"Cycle {cycle.cycle_number}: Confidence {cycle.evaluation.confidence_score:.2f}"
            if cycle.evaluation.missing_aspects:
                eval_info += f", Missing: {', '.join(cycle.evaluation.missing_aspects)}"
            if cycle.web_search_enabled:
                eval_info += f", Web Search: {len(cycle.web_search_results)} results"
            evaluation_insights.append(eval_info)

        insights_text = "\n".join(evaluation_insights)

        try:
            return prompt_manager.render_prompt(
                "final_synthesis",
                question=question,
                answers_text=answers_text,
                references_text=references_text,
                insights_text=insights_text,
            )
        except Exception as e:
            logger.error(f"Failed to render synthesis prompt: {e}")
            return f"Synthesize the following answers for question: {question}\n\n{answers_text}"

    def _create_simple_prompt(self, question: str, context: str) -> str:
        """Create simple RAG prompt using prompt manager"""
        try:
            return prompt_manager.render_prompt(
                "simple_generation", question=question, context=context
            )
        except Exception as e:
            logger.error(f"Failed to render simple prompt: {e}")
            return f"Answer this question based on the context:\n\nQuestion: {question}\n\nContext:\n{context}\n\nAnswer:"

    def _create_insufficient_data_response(
        self, question: str, partial_answers: List[str]
    ) -> str:
        """Create response when insufficient data is available"""
        if partial_answers:
            combined = "\n\n".join(partial_answers)
            return f"""
                Based on the available information in the knowledge base, I can provide the following partial answer:

                {combined}

                However, the knowledge base appears to have insufficient information to fully answer your question: "{question}"

                To get a complete answer, you may need to:
                - Add more relevant documents to the knowledge base
                - Consult additional sources
                - Rephrase your question to focus on available information
                """

        return f"""
                    I apologize, but the knowledge base does not contain sufficient information to answer your question: "{question}"

                    Please consider:
                    - Adding relevant documents to the knowledge base
                    - Checking if your question relates to the available content
                    - Rephrasing your question to match available information
                """

    def _prepare_context(self, documents: List[Document]) -> str:
        """Prepare enhanced context from retrieved documents with rich metadata and token limits"""
        if not documents:
            return "No relevant documents found."

        context_parts = []
        total_tokens = 0
        max_context_tokens = 4000

        for i, doc in enumerate(documents, 1):
            doc_content = self._format_document_for_context(doc, i)
            doc_tokens = self._estimate_tokens(doc_content)

            if total_tokens + doc_tokens > max_context_tokens:
                context_parts.append(
                    f"\n[Additional {len(documents) - i + 1} documents truncated due to length limits]"
                )
                break

            context_parts.append(doc_content)
            total_tokens += doc_tokens

        return "\n\n".join(context_parts)

    def get_memory_stats(self) -> Dict:
        """Get memory cache statistics"""
        if self.memory_cache:
            return self.memory_cache.get_stats()
        return {"cache_disabled": True}

    async def clear_memory_cache(self) -> None:
        """Clear memory cache"""
        logger.info("Clearing memory cache")
        if self.memory_cache:
            self.memory_cache.clear()
            logger.info("Memory cache cleared")
        else:
            logger.warning("Memory cache is disabled, nothing to clear")
