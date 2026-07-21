# tests/test_integration.py
import os
import pytest
from pathlib import Path

from agent.config import Settings
from agent.session import Session, SessionStore
from agent.llm import LLMClient
from agent.runtime import run_turn, build_default_registry, build_memory
from agent.trace import TraceLogger


@pytest.mark.skipif(not os.environ.get("LLM_API_KEY"), reason="LLM_API_KEY not set")
def test_real_weather_question(tmp_path: Path):
    settings = Settings.from_env(sessions_dir=tmp_path)
    settings.context_max_messages = 20
    llm = LLMClient(api_key=settings.llm_api_key, base_url=settings.llm_base_url, model=settings.llm_model)
    session = Session(id="it1")
    registry = build_default_registry()
    memory = build_memory(settings=settings, llm=llm)
    trace = TraceLogger(tmp_path, "it1")
    out = run_turn(session, "What's the weather in Beijing? Use the weather tool.",
                   settings=settings, llm=llm, registry=registry, memory=memory, trace=trace)
    assert isinstance(out, str) and len(out) > 0