"""Phase 1 tests: Starry Coach identity + tech_trend tool.

Validates the transformation from generic agent to learning coach:
- prompts.py contains the coach persona + key behavioural rules
- tech_trend tool is registered, returns structured data, and is swappable
"""
import json
from starry_code.prompts import SYSTEM_PROMPT, EXTRACTOR_PROMPT, NAMING_PROMPT
from starry_code.tools.tech_trend import TechTrendTool, _MockTrendProvider
from starry_code.runtime import build_default_registry
from starry_code.session import Session


# ---- A: prompts ----

def test_system_prompt_contains_coach_identity():
    assert "Starry Coach" in SYSTEM_PROMPT
    assert "项目驱动" in SYSTEM_PROMPT
    assert "tech_trend" in SYSTEM_PROMPT  # 必须提到强制调用的工具


def test_system_prompt_has_must_do_and_must_not():
    assert "必做行为" in SYSTEM_PROMPT
    assert "必不做的行为" in SYSTEM_PROMPT
    # 关键禁止行为：用更稳的子串断言（不依赖完整短语连续匹配）
    assert "编造薪资" in SYSTEM_PROMPT
    assert "不替用户做职业决定" in SYSTEM_PROMPT


def test_extractor_prompt_covers_learning_dimensions():
    for kw in ("技术栈", "学习目标", "项目", "时间投入", "当前水平", "约束"):
        assert kw in EXTRACTOR_PROMPT, f"EXTRACTOR_PROMPT missing dimension: {kw}"


def test_extractor_prompt_format_placeholder():
    assert "{turns}" in EXTRACTOR_PROMPT


def test_naming_prompt_has_learning_bias():
    """NAMING_PROMPT 应该给学习/编程场景明确的命名偏好。"""
    for kw in ("学新技术", "转方向", "求职", "面试", "做项目"):
        assert kw in NAMING_PROMPT, f"NAMING_PROMPT missing scenario: {kw}"


# ---- W1c: few-shot examples in SYSTEM_PROMPT ----

def test_system_prompt_has_few_shot_examples():
    """W1c: SYSTEM_PROMPT contains positive/negative examples showing
    correct vs wrong behavior. Without these, reasoning models sometimes
    bypass the hard rules."""
    assert "正反示例" in SYSTEM_PROMPT, (
        "SYSTEM_PROMPT missing few-shot examples section"
    )
    # The example counter should appear (we shipped 6 pairs)
    assert "示例 6" in SYSTEM_PROMPT


def test_system_prompt_examples_cover_key_behaviors():
    """The 6 example pairs should cover the most common failure modes."""
    expected_topics = [
        "怎么学",         # 例 1: 不要给百科大纲
        "行情",         # 例 2: 必须调 tech_trend
        "update_plan",   # 例 3: 必须调工具改 plan_cache
        "下一步",         # 例 4: 只给 1 件事
        "转",            # 例 5: 不替用户决定
        "大厂",          # 例 6: 不瞎给方案
    ]
    for topic in expected_topics:
        assert topic in SYSTEM_PROMPT, (
            f"SYSTEM_PROMPT few-shot missing topic: {topic}"
        )


def test_system_prompt_examples_have_correct_and_wrong_markers():
    """Each few-shot pair must explicitly mark 错/对 so the LLM sees the
    contrast."""
    assert "❌ 错" in SYSTEM_PROMPT
    assert "✅ 对" in SYSTEM_PROMPT


# ---- B: tech_trend registration ----

def test_tech_trend_registered_in_default_registry():
    reg = build_default_registry()
    assert "tech_trend" in reg.names(), \
        f"expected 'tech_trend' in {reg.names()}"


def test_tech_trend_openai_schema_shape():
    reg = build_default_registry()
    schemas = reg.openai_schemas()
    tech = next(s for s in schemas if s["function"]["name"] == "tech_trend")
    params = tech["function"]["parameters"]
    assert "topic" in params["required"]
    assert "topic" in params["properties"]
    assert "region" in params["properties"]
    assert set(params["properties"]["region"]["enum"]) == {"cn", "global"}


# ---- C: tech_trend execution ----

def test_tech_trend_known_topic_returns_rising_or_stable():
    t = TechTrendTool()
    for topic in ("rust", "go", "python", "react"):
        r = t.execute({"topic": topic}, session=Session(id="x"))
        assert r.ok, f"{topic}: {r.content}"
        data = json.loads(r.content)
        assert data["found"] is True
        assert data["direction"] in ("rising", "stable", "falling")
        assert isinstance(data["demand_score"], int)
        assert 0 <= data["demand_score"] <= 10


def test_tech_trend_unknown_topic_returns_unknown():
    t = TechTrendTool()
    r = t.execute({"topic": "made-up-tech-xyz"}, session=Session(id="x"))
    assert r.ok
    data = json.loads(r.content)
    assert data["found"] is False
    assert data["direction"] == "unknown"
    assert data["demand_score"] is None


def test_tech_trend_missing_topic_arg():
    t = TechTrendTool()
    r = t.execute({}, session=Session(id="x"))
    assert not r.ok
    assert "required" in r.content.lower()


def test_tech_trend_provider_swappable():
    """协议可替换——后续接真实数据源不动工具代码。"""

    class _FakeProvider:
        def fetch(self, topic, region="cn"):
            return {"topic": topic, "found": True, "direction": "rising",
                    "demand_score": 10, "learning_window_weeks": 4,
                    "notes": "from fake", "source": "fake", "region": region}

    original = TechTrendTool._provider
    try:
        TechTrendTool.set_provider(_FakeProvider())
        assert TechTrendTool.current_provider_name() == "_FakeProvider"
        r = TechTrendTool().execute({"topic": "anything"}, session=Session(id="t"))
        assert r.ok
        assert "fake" in r.content
    finally:
        TechTrendTool._provider = original  # 恢复，避免污染其它测试


def test_tech_trend_provider_failure_becomes_tool_error():
    """数据源抛异常时，工具不应当 crash，而是返回 ok=False 的 ToolResult。"""

    class _BrokenProvider:
        def fetch(self, topic, region="cn"):
            raise RuntimeError("upstream 503")

    original = TechTrendTool._provider
    try:
        TechTrendTool.set_provider(_BrokenProvider())
        r = TechTrendTool().execute({"topic": "rust"}, session=Session(id="t"))
        assert not r.ok
        assert "503" in r.content
    finally:
        TechTrendTool._provider = original


# ---- D: 端到端 --mock 跑通（防止 prompt 改坏了把模型整懵）----

def test_mock_pipeline_includes_coach_prompt():
    """build_default_registry 注册了所有工具，prompts 已切到 coach。
    不实际发请求，只验证 import + 装配无误。"""
    from starry_code.runtime import build_memory
    from starry_code.config import Settings
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        s = Settings(sessions_dir=Path(td))
        reg = build_default_registry()
        # Phase 2 added 2 more tools: update_plan, read_artifact → 7 total
        # (calculator, search, todo, weather, tech_trend, update_plan, read_artifact)
        assert len(reg.names()) == 7
        # memory 可以正常装配（mock embedder）
        mem = build_memory(settings=s, llm=None)
        assert mem is not None