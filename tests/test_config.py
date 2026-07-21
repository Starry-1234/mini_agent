import os
from pathlib import Path
from starry_code.config import Settings

def test_defaults(tmp_path, monkeypatch):
    for k in ["LLM_API_KEY", "EMBED_API_KEY", "REDIS_URL", "QDRANT_URL",
              "LLM_BASE_URL", "EMBED_BASE_URL", "LLM_MODEL", "EMBED_MODEL",
              "SHORT_TERM_BACKEND", "VECTOR_BACKEND"]:
        monkeypatch.delenv(k, raising=False)
    s = Settings.from_env(sessions_dir=tmp_path, load_dotenv=False)
    assert s.llm_api_key == ""
    assert s.short_term_backend == "memory"
    assert s.vector_backend == "local"
    assert s.max_tool_iters == 8
    assert s.context_max_messages == 20
    assert s.recent_keep == 8
    assert s.sessions_dir == tmp_path

def test_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_BASE_URL", "https://x.example/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-x")
    monkeypatch.setenv("SHORT_TERM_BACKEND", "redis")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    s = Settings.from_env(sessions_dir=tmp_path, load_dotenv=False)
    assert s.llm_api_key == "sk-test"
    assert s.llm_base_url == "https://x.example/v1"
    assert s.llm_model == "gpt-x"
    assert s.short_term_backend == "redis"
    assert s.redis_url == "redis://localhost:6379/0"
