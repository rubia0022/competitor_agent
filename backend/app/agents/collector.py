"""
Collector Agent：采集证据，输出 RawEvidence[]。
边界：禁止做对比、禁止下结论；只搬运原始信息片段并保留 source_url。
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.agents.base import BaseAgent
from app.providers import SearchProvider
from app.schemas import RawEvidence
from app.storage.trace import TraceStore


COLLECTOR_SYSTEM = """[COLLECTOR]
你是企业竞品分析"信息采集"角色。职责:
1. 仅从外部信息源搜集与竞品有关的原始事实片段；
2. 每条片段必须保留 source_url；
3. 禁止主观判断（如"领先"、"最佳"），禁止对比。

输出 JSON:
{"evidence": [{"competitor","topic","content","source_url","fetched_at","confidence"}]}
"""


class CollectorAgent(BaseAgent):
    role = "collector"

    def __init__(self, llm, trace_store: TraceStore, search: SearchProvider):
        super().__init__(llm, trace_store)
        self.search = search

    def run(
        self,
        task_id: str,
        round_no: int,
        *,
        competitors: List[str],
        industry: str,
        revise_issues: List[Dict[str, Any]] | None = None,
        demo_first_round_drop: tuple[str, ...] = ("persona",),
    ) -> Dict[str, Any]:
        """采集证据。
        demo_first_round_drop: 首轮故意丢弃的 topic 集合，用于演示真实闭环——
        首次采集缺数据像 QA 打回 → 二次采集补齐 → QA 通过。
        生产场景置空即可。
        """
        evidence: List[RawEvidence] = []
        for c in competitors:
            evidence.extend(self.search.search(c, industry))

        # 首轮：模拟“信息不完整”——例如未抓到 persona / pricing
        if round_no == 0 and demo_first_round_drop and not revise_issues:
            evidence = [e for e in evidence if e.topic not in demo_first_round_drop]

        # 闭环：QA 打回时，针对被指出的字段 补齐对应 topic
        if revise_issues:
            missing_topics = _topics_from_issues(revise_issues)
            # 二次采集：重新拉一遍 + 标记为补充
            for c in competitors:
                for ev in self.search.search(c, industry):
                    if ev.topic in missing_topics:
                        evidence.append(
                            RawEvidence(
                                competitor=ev.competitor,
                                topic=ev.topic,
                                content=f"[补充] {ev.content}",
                                source_url=ev.source_url + "#supplement",
                                fetched_at=ev.fetched_at,
                                confidence=min(ev.confidence + 0.1, 1.0),
                            )
                        )

        # trace 写一条 LLM-free 的产物（也走 trace 表，便于回放）
        self.trace.write(
            task_id=task_id,
            agent=self.role,
            round_no=round_no,
            prompt=COLLECTOR_SYSTEM,
            input_payload=f"competitors={competitors}, industry={industry}, revise={bool(revise_issues)}",
            output_payload=f"evidence_count={len(evidence)}",
            tokens_in=0,
            tokens_out=0,
            latency_ms=0,
            status="ok",
        )

        return {"evidence": [e.model_dump(mode="json") for e in evidence]}


def _extract_competitor(field_path: str) -> str | None:
    """从 field_path 中尝试提取竞品名，例如 'competitors["飞书"].pricing[0]' -> '飞书'。"""
    if "[" in field_path and "]" in field_path:
        try:
            return field_path.split("[", 1)[1].split("]", 1)[0]
        except IndexError:
            return None
    return None


def _topics_from_issues(issues: List[Dict[str, Any]]) -> set[str]:
    """从 QA issues 反推缺哪类证据 topic（使用 SchemaConfig 的字段→topic 映射）。"""
    from app.config.schema_config import SchemaConfig
    field_to_topic = SchemaConfig.get().data.get("field_to_topic", {})
    topics: set[str] = set()
    for i in issues:
        p = (i.get("field_path") or "").lower()
        for field, topic in field_to_topic.items():
            if field in p:
                topics.add(topic)
    return topics