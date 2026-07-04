"""
Web search via Tavily. Returns clean, deduped SearchResult objects —
no raw API response leaks past this file.
"""

import logging

from tavily import TavilyClient

from app.config import settings
from app.models import SearchResult, SearchResults

logger = logging.getLogger(__name__)

_client = TavilyClient(api_key=settings.tavily_api_key)


def search(query: str) -> SearchResults:
    """
    Synchronous — Tavily's SDK doesn't expose async, and search is a single
    request (unlike reader.py's N-page fan-out), so this is fine as-is.
    """
    try:
        raw = _client.search(
            query=query,
            max_results=settings.max_search_results,
        )
    except Exception as e:
        logger.warning("search failed for query=%r: %s", query, e)
        return SearchResults(query=query, results=[])

    seen_urls: set[str] = set()
    results: list[SearchResult] = []
    for item in raw.get("results", []):
        url = item.get("url")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        results.append(
            SearchResult(
                title=item.get("title", "").strip() or url,
                url=url,
                snippet=(item.get("content") or "")[:500],
            )
        )

    if len(results) < settings.min_search_results:
        logger.info(
            "search returned only %d results (min=%d) for query=%r",
            len(results), settings.min_search_results, query,
        )

    return SearchResults(query=query, results=results)
