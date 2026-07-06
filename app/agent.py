"""
The agent loop. This is the actual "single agent worker" — everything
else in the app exists to support this function.

Loop: reason (llm.decide_next_action) -> act (tools.execute) -> observe
(append to scratchpad) -> repeat, until finish() or MAX_STEPS.

Failure modes handled explicitly (per the implementation guide's flagged risks):
    - malformed/unparseable tool call from the LLM -> retry once, then force-finish
    - LLM never calls finish() -> force-finish at MAX_STEPS with whatever we have
    - a single tool execution error -> caught by tools.py, never crashes the loop
"""

import logging
import time

from app import llm, memory, summarizer, tools
from app.config import settings
from app.datetime_utils import enrich_query
from app.models import AgentState, AgentStep, PageContent, ResearchResponse, SearchResult, ToolCall, ToolName
from app.prompts import build_system_prompt

logger = logging.getLogger(__name__)


def _successful_read_count(state: AgentState) -> int:
    return len({page.url for page in state.read_pages if page.success and page.text})


def _next_unread_search_url(state: AgentState) -> str | None:
    seen_urls = {page.url for page in state.read_pages}
    for result in state.search_results:
        if result.url not in seen_urls:
            return result.url
    return None


def _redirect_redundant_read(state: AgentState, call: ToolCall) -> ToolCall:
    if call.tool != ToolName.READ:
        return call

    url = call.args.get("url")
    if not url:
        return call

    attempted_urls = {page.url for page in state.read_pages}
    if url not in attempted_urls:
        return call

    replacement_url = _next_unread_search_url(state)
    if not replacement_url or replacement_url == url:
        return call

    logger.info("redirecting duplicate read from %s to next unread result %s", url, replacement_url)
    return ToolCall(
        tool=ToolName.READ,
        args={"url": replacement_url},
        reasoning="Previous URL was already attempted; moving to the next unread source.",
    )


async def run(query: str) -> ResearchResponse:
    enriched_query, was_enriched = enrich_query(query)
    if was_enriched:
        logger.info("enriched query with date context: %r -> %r", query, enriched_query)
    state = AgentState(query=enriched_query, original_query=query, max_steps=settings.max_steps)
    system_prompt = build_system_prompt()
    run_started_at = time.perf_counter()

    while state.step < state.max_steps:
        scratchpad = state.scratchpad_text()
        decision_started_at = time.perf_counter()

        try:
            call = llm.decide_next_action(system_prompt, scratchpad)
        except ValueError as e:
            logger.warning("bad tool call on step %d: %s — retrying once", state.step, e)
            try:
                call = llm.decide_next_action(system_prompt, scratchpad)
            except ValueError as e2:
                logger.error("second bad tool call on step %d: %s — force-finishing", state.step, e2)
                break  # fall through to force-finish path below
        logger.info(
            "step %d decision resolved in %.2fs with tool=%s",
            state.step,
            time.perf_counter() - decision_started_at,
            call.tool.value,
        )
        call = _redirect_redundant_read(state, call)

        if call.tool == ToolName.REFORMULATE:
            tool_started_at = time.perf_counter()
            new_query = call.args.get("query", state.query)
            result = await tools.execute(call)
            state.history.append(AgentStep(step_number=state.step, tool_call=call, tool_result=result))
            state.query = new_query  # scratchpad reflects the new query going forward
            logger.info("step %d tool=%s completed in %.2fs", state.step, call.tool.value, time.perf_counter() - tool_started_at)
            state.step += 1
            continue

        if call.tool == ToolName.FINISH:
            if _successful_read_count(state) < 2:
                next_url = _next_unread_search_url(state)
                if next_url:
                    logger.info("ignoring early finish for query=%r; forcing another read: %s", state.original_query, next_url)
                    forced_call = ToolCall(
                        tool=ToolName.READ,
                        args={"url": next_url},
                        reasoning="Need at least two grounded pages before finishing when available.",
                    )
                    tool_started_at = time.perf_counter()
                    result = await tools.execute(forced_call)
                    state.history.append(AgentStep(step_number=state.step, tool_call=forced_call, tool_result=result))
                    if result.raw:
                        state.read_pages.append(PageContent(**result.raw))
                    logger.info(
                        "step %d forced tool=%s completed in %.2fs",
                        state.step,
                        forced_call.tool.value,
                        time.perf_counter() - tool_started_at,
                    )
                    state.step += 1
                    continue

            tool_started_at = time.perf_counter()
            result = await tools.execute(call)
            state.history.append(AgentStep(step_number=state.step, tool_call=call, tool_result=result))
            state.status = "finished"
            logger.info("step %d tool=%s completed in %.2fs", state.step, call.tool.value, time.perf_counter() - tool_started_at)
            state.step += 1
            break

        tool_started_at = time.perf_counter()
        result = await tools.execute(call)
        state.history.append(AgentStep(step_number=state.step, tool_call=call, tool_result=result))
        logger.info("step %d tool=%s completed in %.2fs", state.step, call.tool.value, time.perf_counter() - tool_started_at)

        if call.tool == ToolName.SEARCH and result.raw:
            state.search_results.extend(SearchResult(**r) for r in result.raw.get("results", []))

        if call.tool == ToolName.READ and result.raw:
            state.read_pages.append(PageContent(**result.raw))

        state.step += 1

    else:
        
        state.status = "max_steps_reached"
        logger.info("query=%r hit max_steps=%d without finish()", state.original_query, state.max_steps)

    if state.status == "running":
        
        state.status = "max_steps_reached"

    summarize_started_at = time.perf_counter()
    final_answer, sources = summarizer.summarize(state)
    state.summary = final_answer
    state.sources = sources
    logger.info("query=%r summarization completed in %.2fs", state.original_query, time.perf_counter() - summarize_started_at)

    note_id = memory.save_note(state, final_answer, sources)
    logger.info("query=%r finished in %.2fs total", state.original_query, time.perf_counter() - run_started_at)

    return ResearchResponse(
        query=state.original_query,
        answer=final_answer,
        sources=sources,
        steps_taken=state.step,
        note_id=note_id,
    )
