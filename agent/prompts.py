# agent/prompts.py
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a helpful Agent. Use tools when needed. "
    "When you have a final answer, reply in plain text (no tool calls). "
    "Keep thoughts brief."
)

EXTRACTOR_PROMPT = (
    "You are a memory extractor. From the recent conversation turn below, "
    "extract 0-5 short, durable facts worth remembering long-term about the user "
    "(preferences, habits, identity, key decisions, constraints). "
    "Output strictly a JSON array of strings. If nothing is worth remembering, "
    "output []. No commentary.\n\nCONVERSATION:\n{turns}\n\nJSON:"
)