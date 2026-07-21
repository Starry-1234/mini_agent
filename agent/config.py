from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(start: Path | None = None) -> None:
    """Load .env from the current directory (and parents) into os.environ.

    Existing env vars take precedence — process env overrides file values.
    Silently no-ops if python-dotenv is not installed (e.g. inside slim images
    where deps are stripped) or if no .env file is found.
    """
    try:
        from dotenv import find_dotenv, load_dotenv  # type: ignore
    except ImportError:
        return
    path = find_dotenv(str(start) if start else ".env", usecwd=True)
    if path:
        load_dotenv(path, override=False)


@dataclass(frozen=True)
class Settings:
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    embed_base_url: str = ""
    embed_api_key: str = ""
    embed_model: str = ""
    short_term_backend: str = "memory"
    vector_backend: str = "local"
    redis_url: str = ""
    qdrant_url: str = ""
    max_tool_iters: int = 8
    context_max_messages: int = 20
    recent_keep: int = 8
    sessions_dir: Path = field(default_factory=lambda: Path("sessions"))

    @classmethod
    def from_env(cls, sessions_dir: Path | None = None, *, load_dotenv: bool = True) -> "Settings":
        if load_dotenv:
            _load_dotenv()
        def get(k, default=""):
            return os.environ.get(k, default)
        def getint(k, default):
            v = os.environ.get(k)
            return int(v) if v else default
        return cls(
            llm_base_url=get("LLM_BASE_URL", "https://api.deepseek.com/v1"),
            llm_api_key=get("LLM_API_KEY"),
            llm_model=get("LLM_MODEL", "deepseek-chat"),
            embed_base_url=get("EMBED_BASE_URL"),
            embed_api_key=get("EMBED_API_KEY"),
            embed_model=get("EMBED_MODEL"),
            short_term_backend=get("SHORT_TERM_BACKEND", "memory"),
            vector_backend=get("VECTOR_BACKEND", "local"),
            redis_url=get("REDIS_URL"),
            qdrant_url=get("QDRANT_URL"),
            max_tool_iters=getint("MAX_TOOL_ITERS", 8),
            context_max_messages=getint("CONTEXT_MAX_MESSAGES", 20),
            recent_keep=getint("RECENT_KEEP", 8),
            sessions_dir=sessions_dir or Path(get("SESSIONS_DIR", "sessions")),
        )
