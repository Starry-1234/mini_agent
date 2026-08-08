# agent/tools/tech_trend.py
from __future__ import annotations
import json
from typing import Protocol
from .base import Tool, ToolResult


class TrendProvider(Protocol):
    """数据源接口。MVP 用 _MockTrendProvider，可替换为 GitHub API / 招聘数据爬虫。

    实现此协议的对象需提供 fetch(topic, region) -> dict，dict 至少包含:
      - topic: str
      - found: bool
      - direction: 'rising' | 'stable' | 'falling' | 'unknown'
      - demand_score: int 0-10 或 None
      - learning_window_weeks: int 或 None
      - notes: str
      - source: str (provider 标识)
    """
    def fetch(self, topic: str, region: str = "cn") -> dict: ...


class _MockTrendProvider:
    """MVP 数据：12 个常见技术的固定曲线。够演示，不够决策。

    数据结构有意设计成「够替换」——真实数据源接入时只需实现 TrendProvider 协议
    并在 build_default_registry() 调用 TechTrendTool.set_provider(...) 换掉实现，
    工具本身不动。
    """

    _DATA = {
        # topic -> (direction, demand_score 0-10, learning_window_weeks, notes)
        "rust":          ("rising",  8, 12, "系统编程+WebAssembly 双线增长，入门曲线陡"),
        "go":            ("rising",  9,  8, "云原生/中间件岗位持续放量，入门平缓"),
        "python":        ("stable",  9,  4, "AI/数据岗位驱动，绝对需求量大但入门者饱和"),
        "typescript":    ("rising",  8,  6, "前端必学，全栈岗位普遍要求"),
        "kubernetes":    ("stable",  8, 10, "运维/后端加分项，但 SRE 岗竞争激烈"),
        "react":         ("stable",  9,  6, "前端主流，生态饱和但岗位基数大"),
        "vue":           ("stable",  7,  4, "国内中小厂主流，与 React 二选一即可"),
        "java":          ("stable",  8,  8, "传统后端基本盘，Spring 全家桶仍主导"),
        "swift":         ("stable",  6, 10, "iOS 岗位收缩，但 AR/VR 可能有回升"),
        "kotlin":        ("stable",  7,  6, "Android 主流，与 Java 二选一"),
        "ai-agent":      ("rising",  7,  8, "LLM 应用爆发期，但岗位要求模糊"),
        "webrtc":        ("stable",  5, 12, "小众但稳定，需要信令/音视频基础"),
    }

    def fetch(self, topic: str, region: str = "cn") -> dict:
        key = topic.lower().strip()
        if key in self._DATA:
            direction, demand, weeks, notes = self._DATA[key]
        else:
            # 未知 topic：返回保守结论，不编造数据
            return {
                "topic": topic,
                "found": False,
                "direction": "unknown",
                "demand_score": None,
                "learning_window_weeks": None,
                "notes": (
                    f"未收录「{topic}」。MVP 阶段建议补充到 _MockTrendProvider._DATA，"
                    "或接入真实数据源。"
                ),
                "source": "mock",
                "region": region,
            }
        return {
            "topic": topic,
            "found": True,
            "direction": direction,
            "demand_score": demand,
            "learning_window_weeks": weeks,
            "notes": notes,
            "source": "mock",
            "region": region,
            "as_of": "2026-08-08",
        }


class TechTrendTool(Tool):
    """查询技术热度、招聘需求趋势、学习时长建议。

    教练在做「该不该学 X」的判断时**必须**调用此工具，而不是凭印象回答。
    """

    # 默认 provider；可在 build_default_registry() 里替换为真实数据源
    _provider: TrendProvider = _MockTrendProvider()

    @classmethod
    def set_provider(cls, provider: TrendProvider) -> None:
        """替换数据源。调用前确保 provider 实现了 TrendProvider 协议。"""
        cls._provider = provider

    @classmethod
    def current_provider_name(cls) -> str:
        return type(cls._provider).__name__

    def __init__(self) -> None:
        super().__init__(
            name="tech_trend",
            description=(
                "查询一个技术/方向的趋势数据：热度方向（rising/stable/falling）、"
                "招聘需求评分（0-10）、0基础到可写进简历的建议学习时长（周）、备注。"
                "当用户问「X 现在行情怎么样」「该不该学 X」「X 还有前景吗」时必须调用。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "技术名/方向，如 'rust'、'kubernetes'、'ai-agent'。小写、英文。",
                    },
                    "region": {
                        "type": "string",
                        "enum": ["cn", "global"],
                        "description": "市场区域，默认 cn",
                    },
                },
                "required": ["topic"],
            },
            execute=self._run,
        )

    def _run(self, args: dict, session) -> ToolResult:
        topic = (args or {}).get("topic", "").strip()
        if not topic:
            return ToolResult.err("topic is required")
        region = (args or {}).get("region", "cn")
        try:
            data = TechTrendTool._provider.fetch(topic, region=region)
        except Exception as e:  # noqa: BLE001
            return ToolResult.err(f"trend provider failed: {type(e).__name__}: {e}")
        # 简洁 JSON 输出（给 LLM 看，不是给人看）
        return ToolResult.ok(json.dumps(data, ensure_ascii=False))