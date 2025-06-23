from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional


class ReflexionDecision(Enum):
    """Reflexion loop decision types"""

    CONTINUE = "continue"
    REFINE_QUERY = "refine_query"
    COMPLETE = "complete"
    INSUFFICIENT_DATA = "insufficient_data"

    def __str__(self) -> str:
        """String representation of the decision"""
        return self.value


class WebSearchStatus(Enum):
    """Web search operation status"""

    SUCCESS = "success"
    FAILED = "failed"
    ERROR = "error"
    DISABLED = "disabled"
    LOW_QUALITY = "low_quality"


@dataclass
class Document:
    """Standard document format across the system"""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None


@dataclass
class WebSearchResult:
    """Web search result with extracted content"""

    url: str
    title: str
    snippet: str
    content: str
    rank: int
    status: WebSearchStatus
    word_count: int = 0
    extraction_strategy: Optional[str] = None
    error_message: Optional[str] = None

    def to_document(self) -> Document:
        """Convert web search result to Document format with content limits"""
        # Truncate content if too long (max 2000 tokens ≈ 8000 chars)
        max_content_length = 8000
        truncated_content = self.content

        if len(self.content) > max_content_length:
            truncated_content = self.content[:max_content_length]
            # Try to break at sentence boundary
            last_period = truncated_content.rfind(".")
            if last_period > max_content_length * 0.8:
                truncated_content = truncated_content[: last_period + 1]
            truncated_content += "\n\n[Content truncated for embedding]"

        return Document(
            content=truncated_content,
            metadata={
                "source": self.url,
                "title": self.title,
                "snippet": self.snippet,
                "rank": self.rank,
                "word_count": self.word_count,
                "extraction_strategy": self.extraction_strategy,
                "search_result": True,
                "status": self.status.value,
                "created_date": datetime.now().isoformat(),
                "file_type": "web_search",
                "file_name": f"web_search_{self.rank}_{self.title[:50]}.txt",
                "content_truncated": len(self.content) > max_content_length,
                "original_length": len(self.content),
            },
            doc_id=f"web_search_{hash(self.url)}",
        )


@dataclass
class StreamingChunk:
    """Streaming Response Chunk"""

    content: str
    is_complete: bool = False
    usage_info: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ReflexionEvaluation:
    """Evaluation result from reflexion assessment"""

    confidence_score: float
    decision: ReflexionDecision
    reasoning: str
    follow_up_queries: List[str]
    covered_aspects: List[str]
    missing_aspects: List[str]
    uncertainty_phrases: List[str]
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ReflexionCycle:
    """Single reflexion cycle data"""

    cycle_number: int
    query: str
    retrieved_docs: List[Document]
    web_search_results: List[WebSearchResult]  # Added this field
    partial_answer: str
    evaluation: ReflexionEvaluation
    timestamp: datetime
    processing_time: float
    web_search_enabled: bool = False  # Added this field

    def __post_init__(self):
        if not hasattr(self, "timestamp"):
            self.timestamp = datetime.now()


@dataclass
class ReflexionMemory:
    """Memory cache for reflexion loops"""

    original_query: str
    cycles: List[ReflexionCycle] = field(default_factory=list)
    final_answer: str = ""
    total_processing_time: float = 0.0
    total_documents_retrieved: int = 0
    total_web_results_retrieved: int = 0  # Added this field

    def add_cycle(self, cycle: ReflexionCycle):
        """Add a reflexion cycle to memory"""
        self.cycles.append(cycle)
        self.total_processing_time += cycle.processing_time
        self.total_documents_retrieved += len(cycle.retrieved_docs)
        self.total_web_results_retrieved += len(cycle.web_search_results)  # Added this

    def get_all_partial_answers(self) -> List[str]:
        """Get all partial answers from cycles"""
        return [cycle.partial_answer for cycle in self.cycles if cycle.partial_answer]

    def get_all_retrieved_docs(self) -> List[Document]:
        """Get all unique retrieved documents"""
        all_docs = []
        seen_ids = set()
        for cycle in self.cycles:
            for doc in cycle.retrieved_docs:
                if doc.doc_id and doc.doc_id not in seen_ids:
                    all_docs.append(doc)
                    seen_ids.add(doc.doc_id)
        return all_docs

    def get_all_web_results(self) -> List[WebSearchResult]:  # Added this method
        """Get all web search results from cycles"""
        all_results = []
        seen_urls = set()
        for cycle in self.cycles:
            for result in cycle.web_search_results:
                if result.url not in seen_urls:
                    all_results.append(result)
                    seen_urls.add(result.url)
        return all_results


class LLMInterface(ABC):
    """Abstract base class for all LLM implementations"""

    @abstractmethod
    def generate_stream(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[StreamingChunk]:
        """Generate text from prompt with streaming"""
        pass

    @abstractmethod
    def chat_stream(
        self, messages: list[Dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[StreamingChunk]:
        """Chat-style interaction with streaming"""
        pass


class ReflexionEvaluatorInterface(ABC):
    """Abstract interface for reflexion evaluation"""

    @abstractmethod
    async def evaluate_response(
        self,
        query: str,
        partial_answer: str,
        retrieved_docs: List[Document],
        cycle_number: int,
    ) -> ReflexionEvaluation:
        """Evaluate if response sufficiently answers the query"""
        pass

    @abstractmethod
    async def generate_follow_up_queries(
        self,
        original_query: str,
        partial_answer: str,
        missing_aspects: List[str],
    ) -> List[str]:
        """Generate follow-up queries based on missing aspects"""
        pass


class VectorStoreInterface(ABC):
    """Abstract base class for all vector store implementations"""

    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents and return their IDs"""
        pass

    @abstractmethod
    async def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        pass

    @abstractmethod
    async def count_documents(self) -> int:
        """Count total number of documents in the vector store"""
        pass

    @abstractmethod
    async def count_web_searches(self) -> int:
        """Count total number of search results in the vector store"""
        pass

    @abstractmethod
    async def delete_all_documents(self, confirm_string: str) -> bool:
        """Delete all documents from the vector store"""
        pass

    @abstractmethod
    async def add_web_search_results(
        self, web_results: List[WebSearchResult]
    ) -> List[str]:
        """Add web search results to the vector store"""
        pass

    @abstractmethod
    async def similarity_search_combined(
        self, query: str, k_docs: int = 3, k_web: int = 2
    ) -> List[Document]:
        """Search both documents and web searches with separate limits"""
        pass


class WebSearchInterface(ABC):
    """Abstract interface for web search implementations"""

    @abstractmethod
    async def search_and_extract(
        self, query: str, num_results: int = 5
    ) -> List[WebSearchResult]:
        """Search web and extract content from results"""
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if web search service is available"""
        pass


class EmbeddingInterface(ABC):
    """Abstract base class for embedding models"""

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text"""
        pass

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts"""
        pass
