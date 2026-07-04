# SIFT: Autonomous AI Research Agent

SIFT is a single-agent autonomous research assistant. Rather than following a fixed retrieval pipeline, it runs a dynamic reasoning loop — Reason, Act, Observe — driven by tool-calling LLMs. At each step, the agent decides for itself whether to search the web, read a specific page, reformulate its query, or conclude and answer.

The project was built as a portfolio piece to demonstrate production-grade agentic backend engineering: an LLM that plans and acts through tool calling, an async web layer that handles real-world failure conditions gracefully, and a layered fallback architecture that keeps the system operational even when individual components (LLM providers, target websites) fail.

SIFT does not use Retrieval-Augmented Generation. There is no vector database, no embeddings, and no similarity search over a pre-built corpus. Every query triggers a live web search, and the agent reasons over freshly fetched content on every run rather than retrieving from an indexed knowledge base. This is a deliberate design choice: the problem being solved is stale LLM knowledge and unverifiable claims, not retrieval over a fixed private document set.

---

## Problem Statement

Pretrained LLMs are frozen at a training cutoff and answer confidently even when their knowledge is outdated, incomplete, or simply wrong. The common workaround — manually searching, opening several tabs, and synthesizing an answer by hand — is slow and doesn't scale. Most automated alternatives replace this with a rigid RAG pipeline that searches once, retrieves once, and answers, regardless of whether that was actually sufficient.

SIFT is built to close that gap: an agent that decides how much research is enough for a given question, gathers information from live sources, and returns an answer that can be traced back to the pages it actually read.

---

## User Case

A user submits a question that requires current, verifiable information. SIFT searches the web, opens the most relevant pages, decides whether it has gathered enough to answer confidently or needs to keep digging, and returns a concise answer with a list of the sources that contributed to it. The interaction happens through a simple web dashboard, so no technical knowledge is required to use the tool, while the underlying API remains available for programmatic use.

---

## High-Level Architecture

```
User Query
    |
    v
FastAPI Endpoint (app/main.py)
    |
    v
Agent Loop Manager (app/agent.py) <------------------+
    |  decides next action                            |
    v                                                  |
LLM Client (app/llm.py)                                |
    |--> Native tool call                              |
    |--> Structured JSON fallback                      |
    |--> Heuristic rule-based fallback                 |
    v  executes tool                                   |
Tool Router (app/tools.py) ----------------------------+
    |  updates history and scratchpad
    |
    |--> search --> Tavily Search API
    |--> read   --> HTTPX async crawler + BeautifulSoup4 parser
    |
    v  loop exits on finish() or max_steps
Summarization Stage (app/summarizer.py)
    - concurrent per-page chunking and summarization
    - multi-document merge into final answer
    |
    v
Memory Layer (app/memory.py) --> data/notes.json
    |
    v
ResearchResponse (answer, sources, step count, note id)
```

The design principle throughout: the agent decides *what to gather*, while summarization, citation tracking, and persistence remain deterministic steps that always run the same way once the loop exits. This keeps the unpredictable part of the system (LLM reasoning) isolated from the parts that need to be reliable (data handling, storage, response formatting).

---

## Reasoning Loop

Each step, the agent reads a scratchpad — a plain-text record of every action taken and its result so far — and selects exactly one of four tools:

| Tool | Behavior |
|---|---|
| `search(query)` | Queries the web via Tavily, returns deduplicated results |
| `read(url)` | Fetches and parses a specific page, strips navigation/scripts/ads, returns clean text |
| `reformulate(query)` | Replaces the working query when search results are sparse or off-target |
| `finish()` | Ends the loop and hands control to summarization |

The loop terminates when the agent calls `finish()`, when it reaches a configured maximum step count, or by forced termination after repeated unrecoverable decision failures. It is designed so that no single failure — a bad model response, a dead link, a rate limit — can leave the run in an unrecoverable or silent-failure state.

---

## Resilience Architecture

This is the part of the system that distinguishes it from a minimal proof of concept.

### Agent decision fallback chain

1. Native tool calling — structured tool calls via the configured loop model, with `tool_choice` forced so the model cannot return plain text mid-loop.
2. Structured JSON completion fallback — if native tool calling fails, the model is prompted to return raw JSON matching the tool schema instead.
3. Heuristic rule-based controller — if every configured LLM is rate-limited or unreachable, a deterministic fallback selects the next action itself (next unread search result, reformulate, or finish), so the agent continues operating even with zero functioning LLM calls.

### Summarization fallback chain

1. Concurrent chunked summarization — long pages are split into overlapping chunks, summarized in parallel, and merged.
2. Extractive snippet fallback — if the summarization model is unavailable, the system deterministically extracts the most relevant sentences from the gathered text and marks the output as a fallback result rather than returning nothing.

### Crawling guardrails

- Minimum grounding constraint: an early `finish()` call is rejected if fewer than two pages have been read and unread search results remain, forcing at least one further read.
- Duplicate-read redirection: a repeated request for an already-read URL is silently redirected to the next unread result instead of wasting a step.
- Bounded concurrency and backoff: page fetches are limited via an async semaphore (default five concurrent requests), with exponential backoff and jitter applied on HTTP 429 responses.

---

## Project Scaffold

```
ai-research-agent/
    app/
        main.py         FastAPI entry point: /research, /notes, /health
        agent.py        The reasoning loop and its guardrails
        llm.py          Dual LLM clients (loop vs summarization), fallback chain
        tools.py        Tool dispatch and duplicate-read redirection
        search.py       Tavily search wrapper, deduplication
        reader.py       Async page fetcher, HTML cleaning, backoff
        summarizer.py   Chunking, concurrent summarization, merging, citations
        memory.py       JSON-backed note persistence
        prompts.py      All prompt templates
        models.py       Shared Pydantic schemas
        config.py       Environment settings and startup validation
        utils.py        Shared helpers
    data/
        notes.json      Persisted research sessions
    tests/
        test_agent.py   Loop and fallback behavior tests
    streamlit_app.py    User-facing dashboard
    .env.example
    requirements.txt
    README.md
```

---

## Tech Stack and Resources Used

**Backend:** FastAPI, Uvicorn, Pydantic and pydantic-settings for typed, validated configuration.

**Agent and LLM layer:** OpenAI-compatible client interface, used identically across OpenRouter, Groq, and local OpenAI-compatible servers (Ollama, LM Studio). Native tool/function calling with a structured JSON and heuristic fallback beneath it.

**Search:** Tavily Search API.

**Web layer:** HTTPX for async HTTP, BeautifulSoup4 with lxml for HTML parsing, asyncio.Semaphore for concurrency control.

**Storage:** Flat-file JSON for research notes and execution traces, chosen deliberately for a single-user local tool; not intended for concurrent write access.

**Frontend:** Streamlit, as a thin client that consumes the FastAPI endpoints over HTTP with no direct dependency on backend internals.

**Testing:** pytest and pytest-asyncio.

**Package management:** uv, for faster dependency resolution and installation than standard pip.

---

## Things Learned

- Designing an actual agent loop — reason, act, observe, repeat — as distinct from a pipeline dressed up as an agent, and recognizing where that distinction actually matters in the code.
- Tool and function calling in practice: schema design, forcing structured output, and handling models that return malformed or hallucinated tool calls without crashing the surrounding system.
- Where single-agent loops silently fail — premature termination, non-termination, and repeated redundant actions — and how to guard against each with explicit constraints rather than hoping the prompt is sufficient.
- Async concurrency patterns for I/O-bound work: bounded concurrency via semaphores, exponential backoff with jitter for rate-limited endpoints, and designing functions that never raise on expected failure conditions.
- Citation design tradeoffs between page-level and sentence-level attribution, and why page-level is the appropriate scope for a first version.
- Building genuine fallback chains rather than a single try/except: multiple independent degradation paths (native call, JSON fallback, heuristic fallback) so the system's behavior degrades predictably rather than catastrophically.
- Practical evaluation of free-tier LLM providers specifically for tool-calling reliability in an agentic context, which is a different evaluation criterion than general benchmark performance or context length.
- Structuring an application so business logic, the LLM layer, and the user interface are fully decoupled, allowing the frontend to be replaced without touching the backend.

---

## Demerits and Current Limitations

- No cross-session memory. Each query is a fully independent run; the agent has no awareness of prior research even though it is logged.
- Citations are page-level, not sentence-level. A source is credited if it contributed to the final answer, without fine-grained claim-to-sentence attribution.
- Storage is a single JSON file, rewritten in full on every save. This is adequate for single-user local use but is not safe under concurrent writers and does not scale past a modest number of stored notes.
- No enforced length constraint on the final answer; output length is emergent from how many pages were read rather than being a deliberate design parameter.
- The heuristic fallback controller, while functional, is necessarily less capable than genuine LLM reasoning — it keeps the system operational during provider outages but produces lower-quality decisions than a working model would.
- Free-tier LLM usage imposes real throughput limits, which constrains how many full research runs are practical per day without paid API access.

---

## Future Possibilities

- Sentence-level citation, mapping specific claims to specific source locations rather than crediting a page as a whole.
- Migration from JSON to SQLite for persistence, enabling concurrent access, indexing, and categorization of past research.
- Multi-agent orchestration, splitting the current single agent into supervised sub-agents with distinct responsibilities such as fact-checking, critique, and retrieval.
- Cross-session memory, allowing the agent to reference prior research when answering follow-up questions. This is treated as a substantial, separate feature rather than an incremental change to the current stateless design.
- Tuning of the query reformulation step, which currently performs a single unguided retry rather than evaluating whether the reformulated query is actually likely to improve results.

---

## Setup and Running

Requirements: Python 3.11 or later, and either uv or standard pip.

```
python -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

cp .env.example .env
# populate LOOP_LLM_API_KEY, SUMMARY_LLM_API_KEY, and TAVILY_API_KEY,
# or set USE_LOCAL_LLM=true to run entirely on a local model server
```

Start the backend:

```
uv run uvicorn app.main:app --reload
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

Start the dashboard:

```
uv run streamlit run streamlit_app.py
```

Available at `http://localhost:8501`.

Run tests:

```
uv run pytest
```