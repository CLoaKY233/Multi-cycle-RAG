from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WebSearchMode(Enum):
    """Web search modes for controlling when web search is triggered"""

    OFF = "off"
    INITIAL_ONLY = "initial_only"
    EVERY_CYCLE = "every_cycle"


class Settings(BaseSettings):
    """
    Application settings with environment variable support
    """

    # SurrealDB Configuration
    surrealdb_url: str = Field(
        default="wss://cloakystores-06a9f7u3jlrsf43q77o8ttu1kk.aws-euw1.surreal.cloud"
    )
    surrealdb_user: str = Field(default="your_username")
    surrealdb_pass: str = Field(default="your_password")
    surrealdb_ns: str = Field(default="rag")
    surrealdb_db: str = Field(default="rag")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # Authentication
    github_token: str = Field(default="")

    # Generation model (primary)
    llm_model: str = Field(default="meta/Meta-Llama-3.1-405B-Instruct")
    llm_temperature: float = Field(default=0.7)
    llm_max_tokens: int = Field(default=1500)  # Reduced from 3000 for faster responses

    # Evaluation model (for self-assessment)
    evaluation_model: str = Field(default="cohere/Cohere-command-r")
    evaluation_temperature: float = Field(default=0.3)
    evaluation_max_tokens: int = Field(default=800)  # Reduced from 1000 for faster evaluation

    # Summary model (for final synthesis)
    summary_model: str = Field(default="meta/Meta-Llama-3.1-70B-Instruct")
    summary_temperature: float = Field(default=0.5)
    summary_max_tokens: int = Field(default=2500)  # Reduced from 4000 for faster synthesis

    # Reflexion configuration
    max_reflexion_cycles: int = Field(default=2)  # Reduced from 3 - most queries don't need 3+
    confidence_threshold: float = Field(default=0.80)  # Reduced from 0.85 - more lenient threshold
    initial_retrieval_k: int = Field(default=5)  # Increased from 3 - better first-pass answers
    reflexion_retrieval_k: int = Field(default=5)

    # Memory cache
    enable_memory_cache: bool = Field(default=True)
    max_cache_size: int = Field(default=100)

    # Embedding
    embedding_model: str = Field(default="text-embedding-3-large")
    embedding_endpoint: str = Field(default="https://models.inference.ai.azure.com")
    embedding_batch_size: int = Field(default=100)

    # Document processing
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)

    # Web Search Configuration
    web_search_mode: WebSearchMode = Field(default=WebSearchMode.OFF)
    web_search_results_count: int = Field(default=5)
    web_search_timeout: int = Field(default=30)
    web_search_min_content_length: int = Field(default=200)
    web_search_max_title_length: int = Field(default=80)

    # Google Search API
    google_api_key: str = Field(default="")
    google_cse_id: str = Field(default="")

    # Web search retrieval
    web_search_retrieval_k: int = Field(default=3)
    web_search_enable_content_extraction: bool = Field(default=True)
    web_crawl_concurrency: int = Field(default=5)  # Concurrent web page crawls (was 2)


# Global settings instance
settings = Settings()
