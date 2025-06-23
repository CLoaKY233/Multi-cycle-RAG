import uuid
from typing import Any, Dict, List

from surrealdb import AsyncSurreal

from ..config.settings import settings
from ..core.exceptions import VectorStoreException
from ..core.interfaces import Document, VectorStoreInterface, WebSearchResult
from ..embeddings.github_embeddings import GithubEmbeddings
from ..utils.logging import logger


class SurrealDBVectorStore(VectorStoreInterface):
    """SurrealDB implementation with native vector search"""

    def __init__(self):
        super().__init__()
        self.client = None
        self.connected = False
        self.embedding_function = GithubEmbeddings()

    async def _ensure_connection(self):
        """Ensure database connection and schema"""
        if self.connected and self.client is not None:
            return

        try:
            # Initialize the client
            self.client = AsyncSurreal(settings.surrealdb_url)

            # Connect and authenticate
            await self.client.connect(settings.surrealdb_url)
            await self.client.signin(
                {
                    "username": settings.surrealdb_user,
                    "password": settings.surrealdb_pass,
                }
            )
            await self.client.use(settings.surrealdb_ns, settings.surrealdb_db)

            # Ensure schema exists
            await self._setup_schema()
            self.connected = True
            logger.info("SurrealDB connection established")

        except Exception as e:
            logger.error(f"SurrealDB connection failed: {str(e)}")
            raise VectorStoreException(f"Connection failed: {str(e)}")

    async def _setup_schema(self):
        """Setup vector storage schema"""
        if not self.client:
            raise VectorStoreException("Client not initialized")

        schema_queries = [
            "DEFINE TABLE IF NOT EXISTS documents SCHEMAFULL;",
            "DEFINE FIELD IF NOT EXISTS content ON documents TYPE string;",
            "DEFINE FIELD IF NOT EXISTS metadata ON documents FLEXIBLE TYPE object;",
            "DEFINE FIELD IF NOT EXISTS embedding ON documents TYPE array<float>;",
            "DEFINE INDEX IF NOT EXISTS hnsw_embedding ON documents FIELDS embedding HNSW DIMENSION 3072 DIST COSINE TYPE F32 EFC 500 M 16;",
            "DEFINE TABLE web_search SCHEMAFULL;",
            "DEFINE FIELD content ON web_search TYPE string;",
            "DEFINE FIELD metadata ON web_search FLEXIBLE TYPE object;",
            "DEFINE FIELD embedding ON web_search TYPE array<float>;",
            "DEFINE INDEX hnsw_embedding ON web_search FIELDS embedding HNSW DIMENSION 3072 DIST COSINE TYPE F32 EFC 500 M 16;",
        ]

        for query in schema_queries:
            try:
                await self.client.query(query)
            except Exception as e:
                logger.error(f"Schema setup error for query '{query}': {e}")
                raise VectorStoreException(f"Schema setup failed: {e}")

    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents with embeddings to SurrealDB"""
        await self._ensure_connection()

        if not documents:
            return []

        if not self.client:
            raise VectorStoreException("Client not connected")

        try:
            # Generate embeddings for all documents
            texts = [doc.content for doc in documents]
            embeddings = await self.embedding_function.embed_documents(texts)

            doc_ids = []

            # Batch insert documents
            for doc, embedding in zip(documents, embeddings):
                doc_id = doc.doc_id or str(uuid.uuid4())

                # Sanitize metadata
                clean_metadata = self._sanitize_metadata(doc.metadata)

                # Create document in SurrealDB
                await self.client.create(
                    "documents",
                    {
                        "id": doc_id,
                        "content": doc.content,
                        "metadata": clean_metadata,
                        "embedding": embedding,
                    },
                )

                doc_ids.append(doc_id)

            logger.info(f"Added {len(doc_ids)} documents to SurrealDB")
            return doc_ids

        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise VectorStoreException(f"Failed to add documents: {str(e)}")

    async def add_web_search_results(
        self, web_results: List[WebSearchResult]
    ) -> List[str]:
        """Add web search result with embeddings to SurrealDB"""
        await self._ensure_connection()

        if not web_results:
            return []

        if not self.client:
            raise VectorStoreException("Client not Connected")

        try:
            # Convert web results to documents
            documents = [result.to_document() for result in web_results]

            # Generate embeddings for all documents
            texts = [doc.content for doc in documents]
            embeddings = await self.embedding_function.embed_documents(texts)
            doc_ids = []

            # Batch insert web search results
            for doc, embedding in zip(documents, embeddings):
                doc_id = doc.doc_id or str(uuid.uuid4())

                # Sanitize metadata
                clean_metadata = self._sanitize_metadata(doc.metadata)
                await self.client.create(
                    "web_search",
                    {
                        "id": doc_id,
                        "content": doc.content,
                        "metadata": clean_metadata,
                        "embedding": embedding,
                    },
                )
                doc_ids.append(doc_id)

            logger.info(f"Added {len(doc_ids)} web search results to SurrealDB")
            return doc_ids

        except Exception as e:
            logger.error(f"Failed to add web search results: {str(e)}")
            raise VectorStoreException(f"Failed to add web search results: {str(e)}")

    async def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform vector similarity search"""
        await self._ensure_connection()

        if not self.client:
            raise VectorStoreException("Client not connected")

        try:
            # Generate query embedding
            query_embedding = await self.embedding_function.embed_text(query)
            query = f"""
            fn::similarity_search({query_embedding}, {k});
            """
            # Perform similarity search using SurrealDB query
            results = await self.client.query(query)

            logger.debug(f"Raw SurrealDB results: {results}")
            # Check if we have results

            if not results or len(results) == 0:
                logger.warning("No results returned from SurrealDB")
                return []

            # Extract results from SurrealDB response
            documents = []
            for result in results:
                if not isinstance(result, dict):
                    continue

                doc = Document(
                    content=result.get("content", ""),
                    metadata={
                        **result.get("metadata", {}),
                        "similarity_score": result.get("score", 0.0),
                    },
                    doc_id=str(result.get("id")),
                )
                documents.append(doc)

            logger.info(f"Retrieved {len(documents)} documents from SurrealDB")
            return documents

        except Exception as e:
            logger.error(f"Similarity search error: {e}")
            return []

    async def similarity_search_combined(
        self, query: str, k_docs: int = 3, k_web: int = 2
    ) -> List[Document]:
        """Perform combined similarity search with proper limits"""
        await self._ensure_connection()

        if not self.client:
            raise VectorStoreException("Client not connected")

        try:
            # Generate query embedding
            query_embedding = await self.embedding_function.embed_text(query)

            # Search documents with limit
            docs_query = f"""
            SELECT id, content, metadata, vector::similarity::cosine(embedding, {query_embedding}) AS score
            FROM documents
            WHERE embedding <|300,COSINE|> {query_embedding}
            ORDER BY score DESC
            LIMIT {k_docs};
            """

            # Search web results with limit
            web_query = f"""
            SELECT id, content, metadata, vector::similarity::cosine(embedding, {query_embedding}) AS score
            FROM web_search
            WHERE embedding <|300,COSINE|> {query_embedding}
            ORDER BY score DESC
            LIMIT {k_web};
            """

            # Execute both searches
            docs_results = await self.client.query(docs_query)
            web_results = await self.client.query(web_query)

            # Process and combine results with token limits
            all_documents = []
            total_tokens = 0
            max_total_tokens = 6000  # Leave room for prompt overhead

            # Process document results first
            if docs_results:
                for result in docs_results:
                    if isinstance(result, dict):
                        content = result.get("content", "")
                        estimated_tokens = len(content) // 4

                        if total_tokens + estimated_tokens > max_total_tokens:
                            break  # Stop adding if we'd exceed limit

                        doc = Document(
                            content=content,
                            metadata={
                                **result.get("metadata", {}),
                                "similarity_score": result.get("score", 0.0),
                                "source_type": "document",
                            },
                            doc_id=str(result.get("id")),
                        )
                        all_documents.append(doc)
                        total_tokens += estimated_tokens

            # Process web search results if we have token budget left
            if web_results and total_tokens < max_total_tokens:
                for result in web_results:
                    if isinstance(result, dict):
                        content = result.get("content", "")
                        estimated_tokens = len(content) // 4

                        if total_tokens + estimated_tokens > max_total_tokens:
                            break  # Stop adding if we'd exceed limit

                        doc = Document(
                            content=content,
                            metadata={
                                **result.get("metadata", {}),
                                "similarity_score": result.get("score", 0.0),
                                "source_type": "web_search",
                            },
                            doc_id=str(result.get("id")),
                        )
                        all_documents.append(doc)
                        total_tokens += estimated_tokens

            # Sort by similarity score
            all_documents.sort(
                key=lambda x: x.metadata.get("similarity_score", 0),
                reverse=True,
            )

            logger.info(
                f"Retrieved {len(all_documents)} combined documents (estimated {total_tokens} tokens)"
            )
            return all_documents

        except Exception as e:
            logger.error(f"Combined similarity search error: {e}")
            return []

    async def count_documents(self) -> int:
        """Get count of documents in the vector store"""
        await self._ensure_connection()

        if not self.client:
            raise VectorStoreException("Client not connected")

        try:
            result = await self.client.query("fn::countdocs()")
            logger.debug(f"Document count result: {result}")

            # Extract the count from the result
            if isinstance(result, int):
                return result
            return 0

        except Exception as e:
            logger.error(f"Failed to count documents: {str(e)}")
            return 0

    async def count_web_searches(self) -> int:
        """Get count of web search results in the vector store"""
        await self._ensure_connection()

        if not self.client:
            raise VectorStoreException("Client not connected")

        try:
            result = await self.client.query("fn::count_web()")
            logger.debug(f"Web search count result: {result}")

            # Extract the count from the result
            if isinstance(result, int):
                return result
            return 0

        except Exception as e:
            logger.error(f"Failed to count web searches: {str(e)}")
            return 0

    async def delete_all_documents(self, confirm_string: str) -> bool:
        """Delete all documents from the vector store"""
        await self._ensure_connection()

        if not self.client:
            raise VectorStoreException("Client not connected")

        try:
            result = await self.client.query(f"fn::deldocs('{confirm_string}')")
            logger.debug(f"Delete all documents ->: {result}")

            if isinstance(result, str):
                success = result.strip().upper() == "DELETED"
                if success:
                    logger.info("All documents deleted successfully")
                else:
                    logger.warning(f"Delete operation returned: {result}")
                return success

            logger.warning("Unexpected response format from delete operation")
            return False

        except Exception as e:
            logger.error(f"Failed to delete all documents: {str(e)}")
            raise VectorStoreException(f"Failed to delete all documents: {str(e)}")

    async def delete_all_web_searches(self, confirm_string: str) -> bool:
        """Delete all web search results from the vector store"""
        await self._ensure_connection()

        if not self.client:
            raise VectorStoreException("Client not connected")

        try:
            result = await self.client.query(f"fn::delweb('{confirm_string}')")
            logger.debug(f"Delete all web searches result: {result}")

            if isinstance(result, str):
                success = result.strip().upper() == "DELETED"
                if success:
                    logger.info("All web search results deleted successfully")
                else:
                    logger.warning(f"Delete operation returned: {result}")
                return success

            logger.warning("Unexpected response format from delete operation")
            return False

        except Exception as e:
            logger.error(f"Failed to delete all web searches: {str(e)}")
            raise VectorStoreException(f"Failed to delete all web searches: {str(e)}")

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata for SurrealDB storage"""
        if not metadata:
            return {}

        sanitized = {}
        for k, v in metadata.items():
            if v is None:
                sanitized[str(k)] = ""
            elif isinstance(v, (str, int, float, bool)):
                sanitized[str(k)] = v
            else:
                sanitized[str(k)] = str(v)

        return sanitized

    async def close(self):
        """Close database connection"""
        if self.client:
            await self.client.close()
            self.connected = False
            logger.info("SurrealDB connection closed")
