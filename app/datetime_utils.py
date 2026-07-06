"""Temporal intent detection and query enrichment with system date context."""

from datetime import datetime

_TEMPORAL_KEYWORDS = [
    "today", "yesterday", "tonight", "this week", "this month", "this year",
    "current", "latest", "upcoming", "now", "as of", "present",
    "recent", "breaking", "just in", "last night", "this morning",
    "this afternoon", "this evening", "right now", "currently",
]


def has_temporal_intent(query: str) -> bool:
    query_lower = query.lower()
    for kw in _TEMPORAL_KEYWORDS:
        if kw in query_lower:
            return True
    return False


def get_date_context() -> str:
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y")


def enrich_query(query: str) -> tuple[str, bool]:
    if has_temporal_intent(query):
        date_str = get_date_context()
        enriched = f"{query} (current date: {date_str})"
        return enriched, True
    return query, False
