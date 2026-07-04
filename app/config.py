"""Runtime configuration. Loaded once as a singleton `settings` object."""

from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_LOOP_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LOOP_MODEL = "openrouter/free"
DEFAULT_SUMMARY_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_SUMMARY_MODEL = "llama-3.1-8b-instant"


def _first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if value:
            return value
    return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None

    loop_llm_api_key: str | None = None
    loop_llm_base_url: str | None = None
    loop_llm_model: str | None = None

    summary_llm_api_key: str | None = None
    summary_llm_base_url: str | None = None
    summary_llm_model: str | None = None

    use_local_llm: bool = False
    local_llm_base_url: str = "http://localhost:11434/v1"
    local_llm_api_key: str = "ollama"
    local_llm_model: str = "llama3.1"

    tavily_api_key: str

    max_steps: int = 5
    min_search_results: int = 3
    max_search_results: int = 8

    fetch_timeout_seconds: int = 7
    max_concurrent_fetches: int = 5

    chunk_size_tokens: int = 2000
    chunk_overlap_tokens: int = 80
    max_pages_to_summarize: int = 2
    max_page_chars_to_summarize: int = 8000

    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 1

    notes_path: str = "data/notes.json"

    log_level: str = "INFO"

    @model_validator(mode="after")
    def _require_provider_keys_unless_local(self) -> "Settings":
        """
        Fail loudly at startup, not mid-run, if you're using a remote
        provider without credentials. Skipped entirely when USE_LOCAL_LLM=true,
        since local servers (Ollama etc.) usually don't need a real key.
        """
        if not self.use_local_llm:
            missing = []
            if not _first_non_empty(self.loop_llm_api_key, self.llm_api_key):
                missing.append("LOOP_LLM_API_KEY or LLM_API_KEY")
            if not _first_non_empty(self.summary_llm_api_key, self.llm_api_key):
                missing.append("SUMMARY_LLM_API_KEY or LLM_API_KEY")
            if missing:
                raise ValueError(
                    f"Missing required env var(s): {', '.join(missing)}. "
                    "Set a shared LLM_API_KEY or the role-specific env vars, "
                    "or set USE_LOCAL_LLM=true to skip remote providers entirely."
                )
        return self

    def loop_llm(self) -> tuple[str, str, str]:
        """Returns (base_url, api_key, model) for the agent loop."""
        if self.use_local_llm:
            return self.local_llm_base_url, self.local_llm_api_key, self.local_llm_model
        return (
            _first_non_empty(self.loop_llm_base_url, self.llm_base_url) or DEFAULT_LOOP_BASE_URL,
            _first_non_empty(self.loop_llm_api_key, self.llm_api_key) or self.llm_api_key or "",
            _first_non_empty(self.loop_llm_model, self.llm_model) or DEFAULT_LOOP_MODEL,
        )

    def summary_llm(self) -> tuple[str, str, str]:
        """Returns (base_url, api_key, model) for summarization/merge."""
        if self.use_local_llm:
            return self.local_llm_base_url, self.local_llm_api_key, self.local_llm_model
        return (
            _first_non_empty(self.summary_llm_base_url, self.llm_base_url) or DEFAULT_SUMMARY_BASE_URL,
            _first_non_empty(self.summary_llm_api_key, self.llm_api_key) or self.llm_api_key or "",
            _first_non_empty(self.summary_llm_model, self.llm_model) or DEFAULT_SUMMARY_MODEL,
        )


@lru_cache
def get_settings() -> Settings:
    """Cached so we don't re-parse .env on every call — import this, not Settings()."""
    return Settings()


settings = get_settings()
