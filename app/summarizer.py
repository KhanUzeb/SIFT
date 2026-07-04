"""
Summarization + citation manager, combined.

Design: chunk-level citations, not per-sentence (per the doc's v1 scope
decision — per-sentence is a stretch goal). Concretely:

    1. Each successfully-read page gets its own summary (attributed to its URL).
    2. Page summaries get merged into one final answer.
    3. Sources list = every page that contributed a summary, in read order.

This keeps citation trivial and correct: if a page's summary made it into
the merge, its URL is a source. No sentence-level attribution to get wrong.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor

from app import llm
from app.config import settings
from app.models import AgentState, PageContent, SourceCitation
from app.prompts import MERGE_SUMMARIES_PROMPT, SUMMARIZE_CHUNK_PROMPT

logger = logging.getLogger(__name__)

# Rough chars-per-token approximation (~4 chars/token for English) —
# good enough for chunking without pulling in a tokenizer dependency.
_CHARS_PER_TOKEN = 4


def _truncate_page_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    clipped = text[:max_chars]
    if " " not in clipped:
        return clipped
    return clipped.rsplit(" ", 1)[0]


def _chunk_text(text: str, chunk_size_tokens: int, overlap_tokens: int) -> list[str]:
    chunk_chars = chunk_size_tokens * _CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * _CHARS_PER_TOKEN

    if len(text) <= chunk_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_chars
        chunks.append(text[start:end])
        start = end - overlap_chars  # step forward, but re-cover the overlap
    return chunks


def _summarize_page(page: PageContent, query: str, chunk_size: int, overlap: int) -> str:
    """One page -> one summary. Chunks internally if the page is long, merges those first."""
    text = _truncate_page_text(page.text or "", settings.max_page_chars_to_summarize)
    chunks = _chunk_text(text, chunk_size, overlap)
    started_at = time.perf_counter()
    logger.info(
        "summarizing page %s in %d chunk(s)",
        page.url,
        len(chunks),
    )

    if len(chunks) == 1:
        summary = llm.generate(
            system_prompt="You summarize webpage content accurately and concisely.",
            user_prompt=SUMMARIZE_CHUNK_PROMPT.format(query=query, chunk=chunks[0]),
        )
        logger.info("summarized page %s in %.2fs", page.url, time.perf_counter() - started_at)
        return summary

    chunk_summaries = [
        llm.generate(
            system_prompt="You summarize webpage content accurately and concisely.",
            user_prompt=SUMMARIZE_CHUNK_PROMPT.format(query=query, chunk=c),
        )
        for c in chunks
    ]
    joined = "\n".join(f"- {s}" for s in chunk_summaries)
    summary = llm.generate(
        system_prompt="You merge partial summaries of the same webpage into one coherent summary.",
        user_prompt=MERGE_SUMMARIES_PROMPT.format(query=query, summaries=joined),
    )
    logger.info("summarized page %s in %.2fs", page.url, time.perf_counter() - started_at)
    return summary


def _summarize_pages_concurrently(pages: list[PageContent], query: str) -> list[str | Exception]:
    max_workers = min(len(pages), settings.max_pages_to_summarize)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _summarize_page,
                page,
                query,
                settings.chunk_size_tokens,
                settings.chunk_overlap_tokens,
            )
            for page in pages
        ]
        results: list[str | Exception] = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                results.append(e)
        return results


def summarize(state: AgentState) -> tuple[str, list[SourceCitation]]:
    """
    Entry point called after the agent loop exits.
    Returns (final_answer_text, sources).
    """
    successful_pages = [p for p in state.read_pages if p.success and p.text]
    successful_pages = successful_pages[: settings.max_pages_to_summarize]
    started_at = time.perf_counter()
    logger.info(
        "building final answer from %d successful page(s)",
        len(successful_pages),
    )

    if not successful_pages:
        logger.warning("no successfully read pages for query=%r — answering with no grounding", state.query)
        return (
            "I couldn't gather enough readable source material to answer this confidently. "
            "The search worked imperfectly this time, or the pages were not accessible.",
            [],
        )

    page_summary_results = _summarize_pages_concurrently(successful_pages, state.original_query)

    page_summaries: list[str] = []
    sources: list[SourceCitation] = []
    for i, (page, result) in enumerate(zip(successful_pages, page_summary_results), 1):
        if isinstance(result, Exception):
            logger.warning("failed to summarize %s: %s", page.url, result)
            continue

        page_summaries.append(f"[{i}] (source: {page.title or page.url})\n{result}")
        sources.append(SourceCitation(title=page.title or page.url, url=page.url))

    if not page_summaries:
        return (
            "I found a few sources, but I couldn't turn them into a reliable summary this time.",
            [],
        )

    joined = "\n\n".join(page_summaries)
    logger.info("merging %d page summary/ies into final answer", len(page_summaries))
    merge_started_at = time.perf_counter()
    final_answer = llm.generate(
        system_prompt="You write clear, well-cited research answers from provided summaries.",
        user_prompt=MERGE_SUMMARIES_PROMPT.format(query=state.original_query, summaries=joined),
    )
    logger.info(
        "merged final answer from %d page summary/ies in %.2fs (total summarize stage %.2fs)",
        len(page_summaries),
        time.perf_counter() - merge_started_at,
        time.perf_counter() - started_at,
    )

    return final_answer, sources
