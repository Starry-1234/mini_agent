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

NAMING_PROMPT = (
    "你是一个会话命名助手。用户刚刚发起了这段对话的第一句话：\n\n"
    "「{user_msg}」\n\n"
    "请为这个对话起一个 2-6 个中文字的简短名字（成语或短语风格，如「天气查询」「周报撰写」）。\n"
    "要求：\n"
    "- 只能用中文汉字、英文字母、数字、下划线、连字符\n"
    "- 不要加标点、不要加引号、不要加任何解释\n"
    "- 直接回复名字本身\n"
)