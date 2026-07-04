"""
Thin wrapper around the LLM API — now with TWO clients, not one:

                        - _loop_client    : used by decide_next_action(), the agent's tool-calling
                        decisions. Configured via settings.loop_llm().
                        _summary_client : used by generate(), plain text summarization/merge.
                        Configured via settings.summary_llm().

Both are OpenAI-compatible clients under the hood — this works for
OpenRouter, Groq, and local servers (Ollama/LM Studio) identically,
since all of them speak the same /v1/chat/completions interface. Which
provider each role actually points at is entirely a .env decision — see
config.py for the routing logic (including the use_local_llm override).

Keep tool-calling parsing here, not in agent.py — agent.py should only
ever see a clean ToolCall / str, never raw SDK response objects.
"""

import json
import logging
from urllib.parse import urlparse

from openai import APIConnectionError, APIStatusError, AuthenticationError, OpenAI

from app.config import settings
from app.models import ToolCall, ToolName

logger = logging.getLogger(__name__)


class LLMServiceError(RuntimeError):
    """Raised when the configured LLM provider cannot complete a request."""


def _make_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def _provider_name(base_url: str) -> str:
    host = urlparse(base_url).netloc or base_url
    if "groq.com" in host:
        return "Groq"
    if "openai.com" in host:
        return "OpenAI"
    if "openrouter.ai" in host:
        return "OpenRouter"
    if "cerebras" in host:
        return "Cerebras"
    if "localhost" in host or "127.0.0.1" in host:
        return "local LLM server"
    return host or "LLM provider"


def _credential_hint(role: str) -> str:
    role_name = role.upper()
    return (
        f"Check {role_name}_LLM_API_KEY / {role_name}_LLM_BASE_URL "
        "or the shared LLM_API_KEY / LLM_BASE_URL."
    )


def _extract_json_object(text: str) -> dict:
    """
    Accept either raw JSON or a fenced markdown block containing JSON.
    """
    candidate = (text or "").strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    return json.loads(candidate)


def _uses_openrouter_free_router(model: str) -> bool:
    return (model or "").strip() == "openrouter/free"


def _is_openrouter_base_url(base_url: str) -> bool:
    return "openrouter.ai" in (urlparse(base_url).netloc or base_url)


def _is_groq_base_url(base_url: str) -> bool:
    return "groq.com" in (urlparse(base_url).netloc or base_url)


def _unique_models(*models: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for model in models:
        candidate = (model or "").strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return ordered


def _model_candidates(role: str) -> list[str]:
    """
    Return a small, ordered model fallback chain for the current provider.
    We keep the first model from settings, then try lighter alternatives if
    the provider rate-limits or rejects the request.
    """
    if role == "loop":
        if _uses_openrouter_free_router(_loop_model):
            # openrouter/free already routes across compatible free models for
            # the requested features, so adding our own free-model chain just
            # creates extra retries without improving the odds.
            return [_loop_model]
        return _unique_models(
            _loop_model,
            "openrouter/free",
        )

    if not _is_openrouter_base_url(_summary_base_url):
        if _is_groq_base_url(_summary_base_url):
            return _unique_models(
                _summary_model,
                "qwen/qwen3-32b",
                "llama-3.1-8b-instant",
            )
        return [_summary_model]

    if _uses_openrouter_free_router(_summary_model):
        return [_summary_model]

    return _unique_models(
        _summary_model,
        "openrouter/free",
    )


def _status_code(exc: Exception) -> int | None:
    return getattr(exc, "status_code", None)


def _is_rate_limited(exc: Exception) -> bool:
    return _status_code(exc) == 429


_loop_base_url, _loop_api_key, _loop_model = settings.loop_llm()
_summary_base_url, _summary_api_key, _summary_model = settings.summary_llm()

_loop_client = _make_client(_loop_base_url, _loop_api_key)
_summary_client = _make_client(_summary_base_url, _summary_api_key)

logger.info("loop LLM: %s @ %s", _loop_model, _loop_base_url)
logger.info("summary LLM: %s @ %s", _summary_model, _summary_base_url)

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web for information relevant to the query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Fetch and read the full content of a webpage by URL. Only call this on URLs returned by search.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reformulate",
            "description": "Replace the current search query with a better one, e.g. if search returned no useful results.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Call this once you have gathered enough grounded information to answer the query.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

FALLBACK_TOOL_CALL_INSTRUCTIONS = """\
Return exactly one next action as JSON.

Allowed tool names:
- search
- read
- reformulate
- finish

Return only valid JSON with this shape:
{
    "tool": "search | read | reformulate | finish",
    "args": {
    "query": "string when needed",
    "url": "string when needed"
    }
}

Rules:
- Do not include markdown fences or explanation.
- Use only one tool.
- If the tool is finish, args must be {}.
- If the tool is read, args must contain only {"url": "..."}.
- If the tool is search or reformulate, args must contain only {"query": "..."}.
"""


def _fallback_decide_next_action(system_prompt: str, scratchpad: str, model: str) -> ToolCall:
    """
    Provider-agnostic fallback for models/providers that reject native
    tool-calling parameters but can still follow structured JSON instructions.
    """
    response = _loop_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": FALLBACK_TOOL_CALL_INSTRUCTIONS},
            {"role": "user", "content": scratchpad},
        ],
    )
    payload = _extract_json_object(response.choices[0].message.content or "")
    try:
        tool_name = ToolName(payload["tool"])
    except (KeyError, ValueError) as e:
        raise ValueError(f"Fallback response returned invalid tool: {payload!r}") from e

    args = payload.get("args", {})
    if not isinstance(args, dict):
        raise ValueError(f"Fallback response returned non-dict args: {payload!r}")

    logger.info("agent decision via json fallback: tool=%s args=%s", tool_name.value, args)
    return ToolCall(tool=tool_name, args=args)


def _heuristic_decide_next_action(scratchpad: str) -> ToolCall:
    """
    Zero-LLM fallback used when the provider is rate-limiting. It keeps the
    agent moving with simple search/read/finish heuristics.
    """
    query = ""
    search_urls: list[str] = []
    read_urls: set[str] = set()

    for line in scratchpad.splitlines():
        if line.startswith("Query: "):
            query = line[len("Query: ") :].strip()
            continue

        if "URL:" in line:
            url = line.split("URL:", 1)[1].strip()
            if url.startswith("http"):
                search_urls.append(url)
            continue

        if line.startswith("  observation: Content from "):
            match = line.split("Content from ", 1)[1].strip()
            url = match.split(" ", 1)[0].strip()
            if url.startswith("http"):
                read_urls.add(url)

    unread_urls = [url for url in search_urls if url not in read_urls]

    if not search_urls and not read_urls:
        logger.info("heuristic loop fallback: issuing search for query=%r", query)
        return ToolCall(tool=ToolName.SEARCH, args={"query": query})

    if unread_urls:
        logger.info("heuristic loop fallback: issuing read for url=%s", unread_urls[0])
        return ToolCall(tool=ToolName.READ, args={"url": unread_urls[0]})

    if read_urls:
        logger.info("heuristic loop fallback: finishing after %d read page(s)", len(read_urls))
        return ToolCall(tool=ToolName.FINISH, args={})

    logger.info("heuristic loop fallback: reformulating query=%r", query)
    return ToolCall(tool=ToolName.REFORMULATE, args={"query": query})


def _request_loop_action(system_prompt: str, scratchpad: str, model: str) -> ToolCall | None:
    try:
        response = _loop_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": scratchpad},
            ],
            tools=TOOL_SCHEMA,
            tool_choice="required",
            max_tokens=128,
        )
    except APIStatusError as e:
        if _is_rate_limited(e):
            logger.warning("loop model %s hit rate limit; trying fallback path", model)
            return None
        if e.status_code == 400:
            logger.warning(
                "native tool calling rejected by %s for model=%s; falling back to structured JSON action mode",
                _provider_name(_loop_base_url),
                model,
            )
            try:
                return _fallback_decide_next_action(system_prompt, scratchpad, model)
            except APIConnectionError as inner:
                raise LLMServiceError(
                    f"Couldn't reach {_provider_name(_loop_base_url)} for the loop model. "
                    "Check the provider URL, firewall, or network access."
                ) from inner
            except AuthenticationError as inner:
                raise LLMServiceError(
                    f"{_provider_name(_loop_base_url)} rejected the loop model credentials. "
                    f"{_credential_hint('loop')}"
                ) from inner
            except APIStatusError as inner:
                raise LLMServiceError(
                    f"{_provider_name(_loop_base_url)} returned an API error for the loop model "
                    f"(status {inner.status_code})."
                ) from inner
        raise LLMServiceError(
            f"{_provider_name(_loop_base_url)} returned an API error for the loop model "
            f"(status {e.status_code})."
        ) from e
    except AuthenticationError as e:
        raise LLMServiceError(
            f"{_provider_name(_loop_base_url)} rejected the loop model credentials. "
            f"{_credential_hint('loop')}"
        ) from e
    except APIConnectionError as e:
        raise LLMServiceError(
            f"Couldn't reach {_provider_name(_loop_base_url)} for the loop model. "
            "Check the provider URL, firewall, or network access."
        ) from e

    message = response.choices[0].message
    if not message.tool_calls:
        raise ValueError(f"Model returned no tool call: {message.content!r}")

    call = message.tool_calls[0]  # single-tool-call-per-turn by design
    try:
        tool_name = ToolName(call.function.name)
    except ValueError as e:
        raise ValueError(f"Model called unknown tool: {call.function.name!r}") from e

    try:
        args = json.loads(call.function.arguments) if call.function.arguments else {}
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned malformed tool args: {call.function.arguments!r}") from e

    logger.info("agent decision: tool=%s args=%s", tool_name.value, args)
    return ToolCall(tool=tool_name, args=args)


def decide_next_action(system_prompt: str, scratchpad: str) -> ToolCall:
    """
    One turn of the agent loop: send the scratchpad, force a tool call back.
    Uses the LOOP client/model — pick a model here for tool-calling
    reliability, not raw quality (see config.py's loop_llm_model).
    Raises ValueError if the model returns something we can't parse into a
    ToolCall — caller (agent.py) decides how to handle that (retry / abort).
    """
    last_error: Exception | None = None
    for model in _model_candidates("loop"):
        try:
            decision = _request_loop_action(system_prompt, scratchpad, model)
            if decision is not None:
                return decision
        except AuthenticationError as e:
            last_error = e
            raise LLMServiceError(
                f"{_provider_name(_loop_base_url)} rejected the loop model credentials. "
                f"{_credential_hint('loop')}"
            ) from e
        except APIConnectionError as e:
            last_error = e
            continue
        except APIStatusError as e:
            last_error = e
            if _is_rate_limited(e) or e.status_code == 400:
                continue
            raise LLMServiceError(
                f"{_provider_name(_loop_base_url)} returned an API error for the loop model "
                f"(status {e.status_code})."
            ) from e
        except ValueError as e:
            last_error = e
            continue

    logger.warning("loop model exhausted fallbacks; using heuristic action selection")
    if last_error and not isinstance(last_error, APIStatusError):
        logger.debug("last loop error before heuristic fallback: %s", last_error)
    return _heuristic_decide_next_action(scratchpad)


def generate(system_prompt: str, user_prompt: str) -> str:
    """
    Plain text generation — used by summarizer.py, no tools involved.
    Uses the SUMMARY client/model — pick a model here for output quality
    (see config.py's summary_llm_model).
    """
    last_error: Exception | None = None
    for model in _model_candidates("summary"):
        try:
            response = _summary_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=512,
            )
            return response.choices[0].message.content or ""
        except AuthenticationError as e:
            last_error = e
            raise LLMServiceError(
                f"{_provider_name(_summary_base_url)} rejected the summary model credentials. "
                f"{_credential_hint('summary')}"
            ) from e
        except APIConnectionError as e:
            last_error = e
            raise LLMServiceError(
                f"Couldn't reach {_provider_name(_summary_base_url)} for the summary model. "
                "Check the provider URL, firewall, or network access."
            ) from e
        except APIStatusError as e:
            last_error = e
            if _is_rate_limited(e) or e.status_code == 400:
                logger.warning("summary model %s hit rate limit; trying fallback path", model)
                continue
            raise LLMServiceError(
                f"{_provider_name(_summary_base_url)} returned an API error for the summary model "
                f"(status {e.status_code})."
            ) from e

    logger.warning("summary model exhausted fallbacks; using extractive fallback summary")
    return _extractive_summary_fallback(user_prompt, last_error)


def _extractive_summary_fallback(user_prompt: str, last_error: Exception | None = None) -> str:
    """
    Deterministic fallback that returns a short extractive summary from the
    provided prompt content. This is better than failing the whole request
    when the summary provider rate-limits.
    """
    text = user_prompt
    marker = "Summaries:"
    if marker in user_prompt:
        text = user_prompt.split(marker, 1)[1].strip()

    lines = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "I found source material, but I couldn't generate a full summary before the model hit a limit."

    snippet = " ".join(lines[:3])
    if len(snippet) > 900:
        snippet = snippet[:900].rsplit(" ", 1)[0]

    if last_error and _status_code(last_error) == 429:
        return snippet + "\n\n[Fallback summary used because the model was rate-limited.]"
    return snippet
