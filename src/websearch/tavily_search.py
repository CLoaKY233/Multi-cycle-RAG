import logging
from typing import List

from tavily import TavilyClient

from src.config.settings import settings
from src.core.interfaces import WebSearchInterface, WebSearchResult, WebSearchStatus

logger = logging.getLogger(__name__)


class TavilyWebSearch(WebSearchInterface):
    """Web search implementation using Tavily API.

    Tavily returns rich search results including content natively,
    so no separate content extraction / crawling step is needed.
    """

    def __init__(self):
        self._client: TavilyClient | None = None

    def _get_client(self) -> TavilyClient | None:
        """Lazy-initialise the Tavily client."""
        if self._client is not None:
            return self._client

        api_key = settings.tavily_api_key
        if not api_key:
            logger.warning(
                "Tavily API key not configured. Set TAVILY_API_KEY in your .env file."
            )
            return None

        try:
            self._client = TavilyClient(api_key=api_key)
            return self._client
        except Exception as exc:
            logger.error("Failed to initialise Tavily client: %s", exc)
            return None

    async def is_available(self) -> bool:
        """Return True if the Tavily client can be initialised."""
        return self._get_client() is not None

    async def search_and_extract(
        self, query: str, num_results: int = 5
    ) -> List[WebSearchResult]:
        """Search with Tavily and return results mapped to WebSearchResult."""
        client = self._get_client()
        if client is None:
            return [
                WebSearchResult(
                    url="",
                    title="",
                    snippet="Tavily API key not configured.",
                    content="",
                    rank=0,
                    status=WebSearchStatus.DISABLED,
                    error_message="TAVILY_API_KEY missing or invalid.",
                )
            ]

        try:
            logger.info("Tavily search: %r (max %d results)", query, num_results)
            response = client.search(
                query=query,
                max_results=num_results,
                include_raw_content=True,
            )
        except Exception as exc:
            logger.error("Tavily search failed: %s", exc)
            return [
                WebSearchResult(
                    url="",
                    title="",
                    snippet=str(exc),
                    content="",
                    rank=0,
                    status=WebSearchStatus.ERROR,
                    error_message=str(exc),
                )
            ]

        results: List[WebSearchResult] = []
        raw_results = response.get("results", [])

        for rank, item in enumerate(raw_results, start=1):
            url: str = item.get("url", "")
            title: str = item.get("title", "")
            snippet: str = item.get("content", "")  # Tavily "content" is the snippet
            # raw_content is the full page text when include_raw_content=True
            full_content: str = item.get("raw_content") or snippet

            word_count = len(full_content.split()) if full_content else 0

            min_length = settings.web_search_min_content_length
            if word_count > 0 and len(full_content) < min_length:
                status = WebSearchStatus.LOW_QUALITY
            else:
                status = WebSearchStatus.SUCCESS

            results.append(
                WebSearchResult(
                    url=url,
                    title=title[: settings.web_search_max_title_length],
                    snippet=snippet,
                    content=full_content,
                    rank=rank,
                    status=status,
                    word_count=word_count,
                    extraction_strategy="tavily_native",
                )
            )
            logger.debug(
                "Result %d: %s (%d words, status=%s)", rank, url, word_count, status
            )

        logger.info("Tavily returned %d results for query %r", len(results), query)
        return results
