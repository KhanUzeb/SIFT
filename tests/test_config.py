from app.config import Settings


def test_role_defaults_split_between_openrouter_and_groq():
    settings = Settings(
        _env_file=None,
        loop_llm_api_key="loop-key",
        summary_llm_api_key="summary-key",
        tavily_api_key="tvly-test",
    )

    assert settings.loop_llm() == (
        "https://openrouter.ai/api/v1",
        "loop-key",
        "openrouter/free",
    )
    assert settings.summary_llm() == (
        "https://api.groq.com/openai/v1",
        "summary-key",
        "llama-3.1-8b-instant",
    )


def test_shared_llm_settings_apply_to_both_roles():
    settings = Settings(
        _env_file=None,
        llm_api_key="shared-key",
        llm_base_url="https://example.ai/v1",
        llm_model="shared-model",
        tavily_api_key="tvly-test",
    )

    assert settings.loop_llm() == (
        "https://example.ai/v1",
        "shared-key",
        "shared-model",
    )
    assert settings.summary_llm() == (
        "https://example.ai/v1",
        "shared-key",
        "shared-model",
    )


def test_role_specific_settings_override_shared_defaults():
    settings = Settings(
        _env_file=None,
        llm_api_key="shared-key",
        llm_base_url="https://example.ai/v1",
        llm_model="shared-model",
        loop_llm_api_key="loop-key",
        loop_llm_base_url="https://loop.example/v1",
        loop_llm_model="loop-model",
        summary_llm_api_key="summary-key",
        summary_llm_base_url="https://summary.example/v1",
        summary_llm_model="summary-model",
        tavily_api_key="tvly-test",
    )

    assert settings.loop_llm() == ("https://loop.example/v1", "loop-key", "loop-model")
    assert settings.summary_llm() == (
        "https://summary.example/v1",
        "summary-key",
        "summary-model",
    )


def test_local_llm_skips_remote_key_validation():
    settings = Settings(
        _env_file=None,
        use_local_llm=True,
        tavily_api_key="tvly-test",
    )

    assert settings.loop_llm() == (
        settings.local_llm_base_url,
        settings.local_llm_api_key,
        settings.local_llm_model,
    )
    assert settings.summary_llm() == (
        settings.local_llm_base_url,
        settings.local_llm_api_key,
        settings.local_llm_model,
    )
