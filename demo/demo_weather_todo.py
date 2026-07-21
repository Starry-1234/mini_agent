"""End-to-end demo: query weather, then add todos. Use --mock for CI / recording without API."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Allow running as `python demo/demo_weather_todo.py` from the project root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.config import Settings
from agent.session import Session
from agent.llm import LLMClient, MockLLMClient
from agent.runtime import run_turn, build_default_registry, build_memory
from agent.trace import TraceLogger


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true")
    p.add_argument("--session", default="demo")
    p.add_argument("--sessions", default="sessions")
    args = p.parse_args()

    settings = Settings.from_env(sessions_dir=Path(args.sessions))
    if args.mock:
        # Script both demo turns. Each turn ends with an extractor call (memory.remember_sid ->
        # extract_facts -> llm.chat), so every turn needs an extra response returning "[]" (no facts).
        # Turn 1: weather tool_call, weather answer, extractor.
        # Turn 2: todo tool_call, todo answer, extractor.
        llm = MockLLMClient(chat_responses=[
            {"choices": [{"message": {"role": "assistant", "content": "checking weather",
                                       "tool_calls": [{"id": "c1", "type": "function",
                                                       "function": {"name": "weather",
                                                                    "arguments": '{"city":"beijing"}'}}]}}]},
            {"choices": [{"message": {"role": "assistant", "content": "Beijing is sunny 26°C."}}]},
            {"choices": [{"message": {"role": "assistant", "content": "[]"}}]},
            {"choices": [{"message": {"role": "assistant", "content": "adding todos",
                                       "tool_calls": [{"id": "c2", "type": "function",
                                                       "function": {"name": "todo",
                                                                    "arguments": '{"action":"add","text":"buy umbrella"}'}}]}}]},
            {"choices": [{"message": {"role": "assistant", "content": "Done. Added 1 todo."}}]},
            {"choices": [{"message": {"role": "assistant", "content": "[]"}}]},
        ])
    else:
        llm = LLMClient(api_key=settings.llm_api_key, base_url=settings.llm_base_url, model=settings.llm_model)

    session = Session(id=args.session)
    registry = build_default_registry()
    memory = build_memory(settings=settings, llm=llm)
    trace = TraceLogger(Path(args.sessions), args.session)

    msgs = [
        "What's the weather in Beijing today?",
        "Great, add a todo for me to buy an umbrella.",
    ]
    for m in msgs:
        print(f"\nUSER: {m}")
        ans = run_turn(session, m, settings=settings, llm=llm, registry=registry, memory=memory, trace=trace)
        print(f"AGENT: {ans}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())