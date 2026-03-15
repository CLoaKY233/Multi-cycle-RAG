from typing import List, Optional

from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential

from src.config.settings import settings
from src.core.exceptions import EmbeddingException
from src.core.interfaces import EmbeddingInterface
from src.utils.logging import logger


class GithubEmbeddings(EmbeddingInterface):
    """Azure AI Inference embeddings implementation"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        token: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.endpoint = endpoint or "https://models.inference.ai.azure.com"
        self.model_name = model_name or "text-embedding-3-large"
        self.token = token or settings.github_token

        if not self.token:
            raise EmbeddingException("GitHub token is required for Azure AI Inference")

        try:
            self.client = EmbeddingsClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.token),
                model=self.model_name,
            )
            logger.info(
                "Azure AI Embeddings initialized",
                model=self.model_name,
                endpoint=self.endpoint,
            )
        except Exception as e:
            raise EmbeddingException(f"Failed to initialize Azure AI embeddings: {e}")

    def _usage_to_dict(self, usage) -> dict:
        """Convert EmbeddingsUsage to dictionary"""
        if not usage:
            return {}
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }

    def _extract_embedding(self, embedding_data) -> List[float]:
        """Extract embedding as List[float] from various formats"""
        if isinstance(embedding_data, list):
            # Already a list of floats
            return embedding_data
        elif isinstance(embedding_data, str):
            # Handle base64 or other string encodings
            raise EmbeddingException(
                "String embeddings not supported. Use 'float' encoding_format."
            )
        else:
            raise EmbeddingException(
                f"Unsupported embedding format: {type(embedding_data)}"
            )

    async def embed_text(self, text: str) -> List[float]:
        """Embed single text using Azure AI Inference"""
        try:
            response = self.client.embed(input=[text])  # type: ignore[attr-defined]

            if not response.data or len(response.data) == 0:
                raise EmbeddingException("No embedding data returned")

            # Extract and validate embedding format
            raw_embedding = response.data[0].embedding
            embedding = self._extract_embedding(raw_embedding)

            logger.debug(
                "Text embedded successfully",
                text_length=len(text),
                embedding_dimension=len(embedding),
                usage=self._usage_to_dict(response.usage),
            )

            return embedding

        except Exception as e:
            logger.error("Text embedding failed", error=str(e), text_length=len(text))
            raise EmbeddingException(f"Text embedding failed: {e}")

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts efficiently using Azure AI Inference"""
        if not texts:
            return []

        try:
            # Azure AI Inference supports batch processing
            batch_size = getattr(settings, "embedding_batch_size", 100)
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                response = self.client.embed(input=batch)  # type: ignore[attr-defined]

                if not response.data:
                    raise EmbeddingException(
                        f"No embedding data returned for batch {i // batch_size + 1}"
                    )

                # Sort by index to maintain order and extract embeddings
                sorted_data = sorted(response.data, key=lambda x: x.index)
                batch_embeddings = [
                    self._extract_embedding(item.embedding) for item in sorted_data
                ]
                all_embeddings.extend(batch_embeddings)

                logger.debug(
                    "Batch embedded successfully",
                    batch_number=i // batch_size + 1,
                    batch_size=len(batch),
                    embedding_dimension=len(batch_embeddings[0])
                    if batch_embeddings
                    else 0,
                    usage=self._usage_to_dict(response.usage),
                )

            logger.info(
                "Documents embedded successfully",
                total_documents=len(texts),
                total_batches=(len(texts) + batch_size - 1) // batch_size,
                embedding_dimension=len(all_embeddings[0]) if all_embeddings else 0,
            )

            return all_embeddings

        except Exception as e:
            logger.error(
                "Document embedding failed",
                error=str(e),
                document_count=len(texts),
            )
            raise EmbeddingException(f"Document embedding failed: {e}")

    def get_embedding_info(self) -> dict:
        """Get embedding model information"""
        return {
            "model_name": self.model_name,
            "endpoint": self.endpoint,
            "provider": "Azure AI Inference",
            "supports_batch": True,
        }
