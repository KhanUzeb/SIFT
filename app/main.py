"""FastAPI entry point. Thin by design — all logic lives in agent.py."""

import logging

from fastapi import FastAPI, HTTPException

from app import memory
from app.agent import run
from app.config import settings
from app.llm import LLMServiceError
from app.models import ResearchRequest, ResearchResponse

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

app = FastAPI(title="SIFT", version="0.1.0")


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest) -> ResearchResponse:
    try:
        return await run(request.query)
    except LLMServiceError as e:
        logger.warning("research blocked by llm provider for query=%r: %s", request.query, e)
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.exception("research run failed for query=%r", request.query)
        raise HTTPException(status_code=500, detail=f"research failed: {e}") from e


@app.get("/notes/{note_id}")
async def get_note(note_id: str):
    note = memory.get_note(note_id)
    if note is None:
        raise HTTPException(status_code=404, detail="note not found")
    return note


@app.get("/notes")
async def list_notes():
    return memory.list_notes()


@app.get("/health")
async def health():
    return {"status": "ok"}
