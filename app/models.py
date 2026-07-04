"""
Pydantic models: request/response schemas + agent state + tool I/O.
Everything else in this app imports from here — keep it dependency-free
(no imports from other app/ modules) to avoid circular imports.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)


class SourceCitation(BaseModel):
    title: str
    url: str


class ResearchResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceCitation]
    steps_taken: int
    note_id: str


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class SearchResults(BaseModel):
    query: str
    results: list[SearchResult] = Field(default_factory=list)


class PageContent(BaseModel):
    url: str
    title: str | None = None
    text: str | None = None
    success: bool
    error: str | None = None


class ToolName(str, Enum):
    SEARCH = "search"
    READ = "read"
    REFORMULATE = "reformulate"
    FINISH = "finish"


class ToolCall(BaseModel):
    """One decision the agent made this step."""
    tool: ToolName
    args: dict = Field(default_factory=dict)
    reasoning: str | None = None  # optional: agent's stated reason, for logging


class ToolResult(BaseModel):
    """What we feed back into the scratchpad after executing a ToolCall."""
    tool: ToolName
    observation: str  # human-readable text the LLM will read back
    raw: dict | None = None  # structured data for our own bookkeeping


class AgentStep(BaseModel):
    step_number: int
    tool_call: ToolCall
    tool_result: ToolResult


class AgentState(BaseModel):
    query: str
    original_query: str
    step: int = 0
    max_steps: int = 8
    status: Literal["running", "finished", "max_steps_reached"] = "running"

    history: list[AgentStep] = Field(default_factory=list)
    search_results: list[SearchResult] = Field(default_factory=list)
    read_pages: list[PageContent] = Field(default_factory=list)

    summary: str | None = None
    sources: list[SourceCitation] = Field(default_factory=list)

    def scratchpad_text(self) -> str:
        """Render history as text the LLM reads back each turn."""
        if not self.history:
            return f"Query: {self.query}\n(no actions taken yet)"
        lines = [f"Query: {self.query}"]
        for h in self.history:
            lines.append(f"[step {h.step_number}] tool={h.tool_call.tool.value} args={h.tool_call.args}")
            lines.append(f"  observation: {h.tool_result.observation}")
        return "\n".join(lines)


class Note(BaseModel):
    note_id: str
    query: str
    summary: str
    sources: list[SourceCitation]
    step_trace: list[AgentStep]
