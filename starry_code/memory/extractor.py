# agent/memory/extractor.py
from __future__ import annotations
import json
import re
from typing import Iterable

from starry_code.llm import LLMClient
from starry_code.prompts import EXTRACTOR_PROMPT


def _format_turns(turns: Iterable[dict]) -> str:
    lines = []
    for t in turns:
        role = t.get("role", "user")
        content = t.get("content", "")
        if isinstance(content, str) and content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) or "(empty)"


def _parse_facts(text: str) -> list[str]:
    if not text:
        return []
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        if isinstance(arr, list):
            return [str(x).strip() for x in arr if str(x).strip()]
    except Exception:
        return []
    return []


def extract_facts(llm: LLMClient, recent_turns: list[dict]) -> list[str]:
    if llm is None or not recent_turns:
        return []
    prompt = EXTRACTOR_PROMPT.format(turns=_format_turns(recent_turns))
    try:
        raw = llm.chat([{"role": "user", "content": prompt}], tools=None)
        text = raw["choices"][0]["message"]["content"] or ""
        return _parse_facts(text)
    except Exception:
        return []