"""
JSON-backed notes storage (V1 — SQLite is a listed future improvement).

One file, one JSON array of Note objects. Fine for a single-user local
tool; not designed for concurrent writers. Fix that when you move to SQLite.
"""

import json
import logging
import uuid
from pathlib import Path

from app.config import settings
from app.models import AgentState, Note, SourceCitation

logger = logging.getLogger(__name__)


def _notes_path() -> Path:
    path = Path(settings.notes_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_all() -> list[dict]:
    path = _notes_path()
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("notes.json is corrupt or unreadable (%s) — starting fresh", e)
        return []


def save_note(state: AgentState, summary: str, sources: list[SourceCitation]) -> str:
    note = Note(
        note_id=str(uuid.uuid4()),
        query=state.original_query,
        summary=summary,
        sources=sources,
        step_trace=state.history,
    )

    notes = _load_all()
    notes.append(json.loads(note.model_dump_json()))

    path = _notes_path()
    with path.open("w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2, ensure_ascii=False)

    logger.info("saved note %s for query=%r", note.note_id, state.original_query)
    return note.note_id


def get_note(note_id: str) -> Note | None:
    for raw in _load_all():
        if raw.get("note_id") == note_id:
            return Note.model_validate(raw)
    return None


def list_notes() -> list[Note]:
    return [Note.model_validate(raw) for raw in _load_all()]
