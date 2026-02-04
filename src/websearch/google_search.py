import asyncio
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, cast

import aiohttp
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from ..config.settings import settings
from ..core.exceptions import (
    WebSearchAPIException,
)
from ..core.interfaces import (
    WebSearchInterface,
    WebSearchResult,
    WebSearchStatus,
)
from ..utils.logging import logger


class ExtractionStrategy(Enum):
    """Content extraction strategies"""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    FALLBACK = "fallback"


@dataclass
class SearchResultData:
    """Raw search result data from Google API"""

    title: str
    url: str
    snippet: str
    rank: int


@dataclass
class ContentSelectors:
    """CSS selectors for different content extraction strategies"""

    primary: str = (
        "article, .article, .post, .content, main, .main-content, .entry-content"
    )
    secondary: str = ".post-content, .article-body, .blog-post, .story-body"
    fallback: str = 'p, .text, .description, .summary, div[class*="content"]'


class GoogleWebSearch(WebSearchInterface):
    """Google Custom Search API integration with content extraction"""

    def __init__(self):
        """Initialize Google web search"""
        self.api_key = settings.google_api_key
        self.cse_id = settings.google_cse_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"

        # Async HTTP session (reused for all requests)
        self._session: Optional[aiohttp.ClientSession] = None

        self.selectors = ContentSelectors()

        # Validate configuration
        if not self.api_key or not self.cse_id:
            logger.warning("Google Search API credentials not configured")

    async def _ensure_session(self):
        """Ensure aiohttp session is initialized"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": "RAG-WebSearch/1.0 (Compatible; Educational)"}
            )

    async def is_available(self) -> bool:
        """Check if web search service is available"""
        return bool(self.api_key and self.cse_id)

    async def search_and_extract(
        self, query: str, num_results: int = 5
    ) -> List[WebSearchResult]:
        """Search web and extract content from results"""
        if not await self.is_available():
            logger.warning("Web search not available - missing API credentials")
            return []

        try:
            # Perform Google search
            search_results = await self._perform_search(query, num_results)
            if not search_results:
                logger.warning("No search results returned")
                return []

            # Extract content from search results
            web_results = await self._extract_content_from_results(search_results)

            successful_count = len(
                [r for r in web_results if r.status == WebSearchStatus.SUCCESS]
            )
            logger.info(
                f"Web search completed: {successful_count}/{len(web_results)} successful extractions",
                query=query,
            )

            return web_results

        except Exception as e:
            logger.error(f"Web search failed: {e}", query=query)
            return []

    async def _perform_search(
        self, query: str, num_results: int
    ) -> List[SearchResultData]:
        """Perform Google Custom Search with rate limiting (async)"""
        await self._ensure_session()

        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "num": min(num_results, settings.web_search_results_count),
        }

        try:
            # Add retry logic for rate limiting
            max_retries = 3
            for attempt in range(max_retries):
                async with self._session.get(
                    self.base_url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=settings.web_search_timeout),
                ) as response:
                    if response.status == 429:  # Rate limited
                        wait_time = 2**attempt  # Exponential backoff
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                        continue

                    response.raise_for_status()
                    data = await response.json()

                    # Check for API errors
                    if "error" in data:
                        raise WebSearchAPIException(f"Google API Error: {data['error']}")

                    items = data.get("items", [])
                    logger.info(f"Google search returned {len(items)} results", query=query)

                    # Convert to SearchResultData objects
                    search_results = []
                    for i, item in enumerate(items, 1):
                        search_results.append(
                            SearchResultData(
                                title=item.get("title", ""),
                                url=item.get("link", ""),
                                snippet=item.get("snippet", ""),
                                rank=i,
                            )
                        )

                    return search_results
            else:
                raise WebSearchAPIException("Max retries exceeded due to rate limiting")

        except aiohttp.ClientError as e:
            raise WebSearchAPIException(f"Search API request failed: {e}")
        except Exception as e:
            raise WebSearchAPIException(f"Search failed: {e}")

    async def _extract_content_from_results(
        self, search_results: List[SearchResultData]
    ) -> List[WebSearchResult]:
        """Extract content from search result URLs"""
        web_results = []

        if not settings.web_search_enable_content_extraction:
            # Return results with just snippets if content extraction is disabled
            for result in search_results:
                web_results.append(
                    WebSearchResult(
                        url=result.url,
                        title=result.title,
                        snippet=result.snippet,
                        content=result.snippet,  # Use snippet as content
                        rank=result.rank,
                        status=WebSearchStatus.SUCCESS,
                        word_count=len(result.snippet.split()),
                        extraction_strategy="snippet_only",
                    )
                )
            return web_results

        # Extract content using Crawl4AI v0.6.x - CORRECTED VERSION
        browser_config = BrowserConfig(
            headless=True,
            browser_type="chromium",  # Explicitly set browser type
            verbose=False,
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Process URLs concurrently with configurable limits
            semaphore = asyncio.Semaphore(
                settings.web_crawl_concurrency
            )  # Increased from 2 to 5 for faster extraction

            tasks = [
                self._extract_single_url_fixed(crawler, result, semaphore)
                for result in search_results
            ]

            web_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and convert to proper results
            final_results = []
            for i, result in enumerate(web_results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Content extraction failed for {search_results[i].url}: {result}"
                    )
                    # Create fallback result with snippet
                    final_results.append(
                        WebSearchResult(
                            url=search_results[i].url,
                            title=search_results[i].title,
                            snippet=search_results[i].snippet,
                            content=search_results[i].snippet,
                            rank=search_results[i].rank,
                            status=WebSearchStatus.FAILED,
                            error_message=str(result),
                        )
                    )
                else:
                    final_results.append(result)

        return final_results

    async def _extract_single_url_fixed(
        self,
        crawler: AsyncWebCrawler,
        search_result: SearchResultData,
        semaphore: asyncio.Semaphore,
    ) -> WebSearchResult:
        """Extract content from a single URL - FIXED VERSION"""
        async with semaphore:
            try:
                # CORRECTED: Configure crawler for v0.6.x API
                run_config = CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    markdown_generator=DefaultMarkdownGenerator(
                        content_filter=PruningContentFilter(
                            threshold=0.55,
                            threshold_type="fixed",
                            min_word_threshold=50,
                        )
                    ),
                    css_selector=self.selectors.primary,
                    excluded_tags=[
                        "nav",
                        "header",
                        "footer",
                        "aside",
                        "form",
                        "script",
                        "style",
                        "menu",
                        "sidebar",
                    ],
                    word_count_threshold=25,
                    exclude_external_links=True,
                    exclude_social_media_links=True,
                    remove_overlay_elements=True,
                )

                # CORRECTED: Crawl the URL with proper error handling
                result = await crawler.arun(url=search_result.url, config=run_config)

                # CORRECTED: Robust content extraction
                content = await self._extract_content_with_strategies_fixed(result)

                if self._validate_content(content, search_result.url):
                    status = WebSearchStatus.SUCCESS
                    strategy = "content_extraction"
                else:
                    # Use snippet as fallback
                    content = search_result.snippet
                    status = WebSearchStatus.LOW_QUALITY
                    strategy = "snippet_fallback"

                # Clean and truncate title
                title = self._clean_title(search_result.title)

                return WebSearchResult(
                    url=search_result.url,
                    title=title,
                    snippet=search_result.snippet,
                    content=content,
                    rank=search_result.rank,
                    status=status,
                    word_count=len(content.split()) if content else 0,
                    extraction_strategy=strategy,
                )

            except Exception as e:
                logger.error(f"Content extraction failed for {search_result.url}: {e}")
                return WebSearchResult(
                    url=search_result.url,
                    title=search_result.title,
                    snippet=search_result.snippet,
                    content=search_result.snippet,  # Fallback to snippet
                    rank=search_result.rank,
                    status=WebSearchStatus.ERROR,
                    error_message=str(e),
                )

    async def _extract_content_with_strategies_fixed(self, crawl_result) -> str:
        """Extract content using multiple strategies - COMPLETELY FIXED"""

        # Strategy 1: Try markdown with multiple access patterns
        try:
            if hasattr(crawl_result, "markdown") and crawl_result.markdown:
                # Cast to Any to handle type checker issues
                markdown_obj: Any = cast(Any, crawl_result.markdown)

                # Try different markdown access patterns based on working example
                content = ""
                if hasattr(markdown_obj, "fit_markdown") and markdown_obj.fit_markdown:
                    content = str(markdown_obj.fit_markdown).strip()
                elif (
                    hasattr(markdown_obj, "raw_markdown") and markdown_obj.raw_markdown
                ):
                    content = str(markdown_obj.raw_markdown).strip()
                elif markdown_obj:
                    content = str(markdown_obj).strip()

                if content and len(content) >= settings.web_search_min_content_length:
                    cleaned_content = self._clean_content_advanced(content)
                    if (
                        cleaned_content
                        and len(cleaned_content)
                        >= settings.web_search_min_content_length
                    ):
                        return cleaned_content
        except Exception as e:
            logger.debug(f"Markdown extraction failed: {e}")

        # Strategy 2: Try cleaned HTML
        try:
            if hasattr(crawl_result, "cleaned_html") and crawl_result.cleaned_html:
                content = self._html_to_text(crawl_result.cleaned_html)
                if content and len(content) >= settings.web_search_min_content_length:
                    return content
        except Exception as e:
            logger.debug(f"Cleaned HTML extraction failed: {e}")

        # Strategy 3: Raw HTML as last resort
        try:
            if hasattr(crawl_result, "html") and crawl_result.html:
                content = self._html_to_text(crawl_result.html)
                if content and len(content) >= settings.web_search_min_content_length:
                    return content
        except Exception as e:
            logger.debug(f"Raw HTML extraction failed: {e}")

        return ""

    def _clean_content_advanced(self, content: str) -> str:
        """Advanced content cleaning based on working example"""
        if not content:
            return content

        # Navigation patterns to remove (from working example)
        nav_patterns = [
            r"Skip to main content.*?\n",
            r"Open menu.*?\n",
            r"Log [Ii]n.*?\n",
            r"Sign [Uu]p.*?\n",
            r"Subscribe.*?\n",
            r"Follow us.*?\n",
            r"Get App.*?\n",
            r"Download.*?\n",
            r"\[.*?\]\(#.*?\)",  # Skip links
            r"^\s*\*\s*&nbsp;\s*$",  # Empty navigation items
            r"^\s*\*\s*$",  # Empty bullet points
        ]

        cleaned = content
        for pattern in nav_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.MULTILINE | re.IGNORECASE)

        # Remove excessive whitespace
        cleaned = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned)
        cleaned = re.sub(r"^\s+|\s+$", "", cleaned, flags=re.MULTILINE)

        # Filter navigation lines
        lines = cleaned.split("\n")
        filtered_lines = []

        nav_indicators = [
            "menu",
            "navigation",
            "log in",
            "sign up",
            "subscribe",
            "follow",
            "get app",
        ]

        for line in lines:
            line_lower = line.lower().strip()
            if not line_lower:
                filtered_lines.append(line)
                continue

            # Skip obvious navigation lines
            is_nav = (
                any(indicator in line_lower for indicator in nav_indicators)
                and len(line.strip()) < 50
            )

            if not is_nav:
                filtered_lines.append(line)

        return "\n".join(filtered_lines).strip()

    def _validate_content(self, content: str, url: str) -> bool:
        """Validate extracted content quality"""
        if not content or len(content.strip()) < settings.web_search_min_content_length:
            return False

        # Check for common error pages
        error_indicators = [
            "404 not found",
            "access denied",
            "page not found",
            "error occurred",
            "temporarily unavailable",
        ]

        content_lower = content.lower()
        if any(indicator in content_lower for indicator in error_indicators):
            logger.warning(f"Error page detected for {url}")
            return False

        # Check content-to-noise ratio
        words = content.split()
        if len(words) < 50:  # Too short
            return False

        return True

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to clean text"""
        # Remove script and style elements
        html = re.sub(
            r"<script[^>]*>.*?</script>",
            "",
            html,
            flags=re.DOTALL | re.IGNORECASE,
        )
        html = re.sub(
            r"<style[^>]*>.*?</style>",
            "",
            html,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Remove HTML tags
        html = re.sub(r"<[^>]+>", "", html)

        # Clean up whitespace
        html = re.sub(r"\s+", " ", html)
        html = html.strip()

        return html

    def _clean_title(self, title: str) -> str:
        """Clean and truncate title"""
        # Remove common title suffixes
        title = re.sub(r"\s*[-|]\s*.*$", "", title)

        # Truncate if too long
        if len(title) > settings.web_search_max_title_length:
            title = (
                title[: settings.web_search_max_title_length].rsplit(" ", 1)[0] + "..."
            )

        return title.strip()

    async def close(self):
        """Clean up resources"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    def _validate_content_for_storage(self, content: str, url: str) -> bool:
        """Validate content before storing in database"""
        if not content or len(content.strip()) < settings.web_search_min_content_length:
            return False

        # Token limit check (rough estimate: 1 token ≈ 4 characters)
        estimated_tokens = len(content) // 4
        if estimated_tokens > 2000:  # Limit to 2000 tokens per document
            return False

        # Quality checks from working example
        words = len(content.split())
        if words < 50:  # Too short
            return False

        # Check word-to-link ratio
        links = content.count("[")  # Markdown links
        if words > 0 and links / words > 0.3:  # Too many links
            return False

        # Check for meaningful sentences
        sentences = content.split(".")
        meaningful_sentences = [s for s in sentences if len(s.strip()) > 20]
        if len(meaningful_sentences) < 3:
            return False

        return True

    def _truncate_content_for_storage(
        self, content: str, max_tokens: int = 2000
    ) -> str:
        """Truncate content to fit token limits"""
        # Rough estimate: 1 token ≈ 4 characters
        max_chars = max_tokens * 4

        if len(content) <= max_chars:
            return content

        # Truncate at sentence boundary
        truncated = content[:max_chars]
        last_period = truncated.rfind(".")
        if last_period > max_chars * 0.8:  # If we can find a good break point
            truncated = truncated[: last_period + 1]

        return truncated + "\n\n[Content truncated for storage]"
