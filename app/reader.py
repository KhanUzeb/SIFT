"""
Async webpage fetcher + HTML cleaner.

Design constraints from the implementation guide:
- concurrent fetches, bounded by a semaphore (don't hammer hosts)
- per-request timeout
- 429 / rate-limit aware backoff, not just a blind retry
- a single bad page must never crash the batch — always return a
  PageContent with success=False rather than raising

Called two ways:
- fetch_one(url): used by the agent loop's `read` tool (single page, on demand)
- fetch_many(urls): used if you ever want to batch-read search results directly
"""

import asyncio
import logging
import random

import httpx
from bs4 import BeautifulSoup

from app.config import settings
from app.models import PageContent

logger = logging.getLogger(__name__)

_semaphore = asyncio.Semaphore(settings.max_concurrent_fetches)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ResearchAgent/0.1; "
        "+https://example.com/bot)"
    )
}

_MAX_RETRIES = 3
_BASE_BACKOFF_SECONDS = 1.0
_NON_RATE_LIMIT_BACKOFF_SECONDS = 0.35


def _clean_text(html: str) -> tuple[str | None, str]:
    """Strip nav/script/ads, return (title, clean_text)."""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "iframe"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else None

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    clean = "\n".join(line for line in lines if line)

    return title, clean


async def _fetch_with_retries(client: httpx.AsyncClient, url: str) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = await client.get(
                url,
                headers=_HEADERS,
                timeout=settings.fetch_timeout_seconds,
                follow_redirects=True,
            )
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else _BASE_BACKOFF_SECONDS * (2 ** attempt)
                delay += random.uniform(0, 0.5)  # jitter, avoid thundering herd
                logger.info("429 from %s, backing off %.1fs (attempt %d)", url, delay, attempt + 1)
                await asyncio.sleep(delay)
                continue
            resp.raise_for_status()
            return resp
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as e:
            last_exc = e
            if attempt < _MAX_RETRIES - 1:
                delay = _NON_RATE_LIMIT_BACKOFF_SECONDS * (attempt + 1) + random.uniform(0, 0.2)
                await asyncio.sleep(delay)
    raise last_exc or RuntimeError(f"fetch failed for {url} with no exception captured")


async def fetch_one(url: str) -> PageContent:
    """Fetch + parse a single page. Never raises — failures come back as success=False."""
    async with _semaphore:
        try:
            async with httpx.AsyncClient() as client:
                resp = await _fetch_with_retries(client, url)
        except Exception as e:
            logger.warning("failed to fetch %s: %s", url, e)
            return PageContent(url=url, success=False, error=str(e))

    content_type = resp.headers.get("content-type", "")
    if "text/html" not in content_type:
        return PageContent(url=url, success=False, error=f"unsupported content-type: {content_type}")

    try:
        title, text = _clean_text(resp.text)
    except Exception as e:
        logger.warning("failed to parse %s: %s", url, e)
        return PageContent(url=url, success=False, error=f"parse error: {e}")

    if not text or len(text) < 100:
        return PageContent(url=url, success=False, error="page returned negligible content")

    return PageContent(url=url, title=title, text=text, success=True)


async def fetch_many(urls: list[str]) -> list[PageContent]:
    """Concurrent fetch of multiple URLs — semaphore inside fetch_one bounds concurrency."""
    return await asyncio.gather(*(fetch_one(u) for u in urls))
