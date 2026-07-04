"""
Dispatches a ToolCall to the right implementation and returns a ToolResult.

agent.py should never call search.py / reader.py directly — everything
goes through here so there's one place that turns raw data into the
"observation" text the LLM reads back.
"""

import asyncio
import logging

from app import reader, search
from app.models import ToolCall, ToolName, ToolResult

logger = logging.getLogger(__name__)


def _format_search_observation(results) -> str:
    if not results.results:
        return f"No useful results found for '{results.query}'. Consider trying a clearer or narrower query."
    lines = [f"Found {len(results.results)} results for '{results.query}':"]
    for i, r in enumerate(results.results, 1):
        lines.append(f"{i}. {r.title}\n   URL: {r.url}\n   {r.snippet}")
    return "\n".join(lines)


def _format_read_observation(page) -> str:
    if not page.success:
        return f"Couldn't read {page.url}: {page.error}. Try a different source."
    preview = page.text[:3000]
    truncated = "... [truncated]" if len(page.text) > 3000 else ""
    return f"Content from {page.url} ({page.title or 'untitled'}):\n{preview}{truncated}"


async def execute(call: ToolCall) -> ToolResult:
    if call.tool == ToolName.SEARCH:
        query = call.args.get("query", "")
        results = await asyncio.to_thread(search.search, query)
        return ToolResult(
            tool=call.tool,
            observation=_format_search_observation(results),
            raw=results.model_dump(),
        )

    if call.tool == ToolName.READ:
        url = call.args.get("url", "")
        page = await reader.fetch_one(url)
        return ToolResult(
            tool=call.tool,
            observation=_format_read_observation(page),
            raw=page.model_dump(),
        )

    if call.tool == ToolName.REFORMULATE:
        new_query = call.args.get("query", "")
        return ToolResult(
            tool=call.tool,
            observation=f"Updated the search query to: '{new_query}'",
            raw={"new_query": new_query},
        )

    if call.tool == ToolName.FINISH:
        return ToolResult(tool=call.tool, observation="Enough information gathered to draft the answer.")

    raise ValueError(f"No dispatcher registered for tool: {call.tool}")
