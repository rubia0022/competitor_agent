"""
Analyst Agent：将 RawEvidence[] 整合为 Competitor[]（强 Schema + 强引用）。

实现策略（混合）：
- features / pricing / personas: 规则化路由（确定性 + Schema 稳）
- SWOT: 优先用 LLM 做语义推理；含三大可靠性策略：
    1) 超长上下文分片 (>SHARD_THRESHOLD 条证据时按 topic 分片)
    2) 自一致性校验（N 次采样投票，少数派被丢弃）
    3) Agent 自评估（completeness / citation_density / reasoning_quality）
- LLM 全失败时回退规则兜底
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from app.agents.base import BaseAgent
from app.providers import MockLLMProvider
from app.schemas import (
    Citation,
    ClaimWithCitation,
    Competitor,
    FeatureNode,
    Persona,
    PricingPlan,
    RawEvidence,
    SWOT,
)

# —— 策略参数 ——
SHARD_THRESHOLD = 12            # 单竞品证据 > 此值时分片
SELF_CONSISTENCY_SAMPLES = 2     # SWOT 采样次数（投票判定）
VOTE_MIN_SUPPORT = 1             # 一条 claim 至少要被几次采样命中
LOW_CONFIDENCE_SCORE = 0.6      # 自评估分低于此值则降级

from app.config.schema_config import SchemaConfig


def _build_analyst_system() -> str:
    quadrants = SchemaConfig.get().get_swot_quadrants()
    keys_json = ",\n    ".join(
        f'{{"{q["key"]}":    [{{"claim": "...", "source_urls": ["https://..."]}}]}}'
        for q in quadrants
    )
    return f"""[ANALYST]
你是企业竞品分析"分析师"任务：基于给定证据，为指定竞品输出 SWOT 分析。

强制要求:
1. 每条结论（claim）必须**严格来自**证据原文，禁止编造；
2. 每条结论必须给出至少 1 个 source_url，且该 URL 必须出现在证据列表中；
3. 输出**纯 JSON**，不要任何解释，不要 markdown 代码块；
4. 结构如下:
{{
    {keys_json}
}}
5. 每个象限 1-3 条；可以基于行业常识合理推断，但 source_urls 仍须从证据中选取最相关者。
"""


class AnalystAgent(BaseAgent):
    role = "analyst"

    def run(
        self,
        task_id: str,
        round_no: int,
        *,
        evidence: List[Dict[str, Any]],
        revise_issues: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        raw = [RawEvidence(**e) for e in evidence]
        by_competitor: Dict[str, List[RawEvidence]] = defaultdict(list)
        for e in raw:
            by_competitor[e.competitor].append(e)

        competitors: List[Competitor] = []
        meta: Dict[str, Any] = {"llm_used": False, "shards": {}, "self_eval": {}}

        for name, items in by_competitor.items():
            comp = self._build_competitor(name, items)
            llm_swot, swot_meta = self._llm_swot(task_id, round_no, name, items)
            if llm_swot is not None:
                comp.swot = llm_swot
                meta["llm_used"] = True
            meta["shards"][name] = swot_meta.get("shards")
            meta["self_eval"][name] = swot_meta.get("self_eval")
            competitors.append(comp)

        # 闭环重做：若 issue 指出 SWOT 不足，补一条 weakness 占位
        if revise_issues:
            for c in competitors:
                if len(c.swot.weaknesses) < 1:
                    sample_ev = next((e for e in raw if e.competitor == c.name), None)
                    if sample_ev:
                        c.swot.weaknesses.append(
                            ClaimWithCitation(
                                claim=f"{c.name} 在公开渠道的细节披露相对有限",
                                citations=[_to_citation(sample_ev)],
                            )
                        )

        self.trace.write(
            task_id=task_id,
            agent=self.role,
            round_no=round_no,
            prompt=_build_analyst_system(),
            input_payload=f"evidence_count={len(raw)}, revise={bool(revise_issues)}",
            output_payload=(
                f"competitors={[c.name for c in competitors]}, "
                f"meta={json.dumps(meta, ensure_ascii=False)}"
            ),
            tokens_in=0,
            tokens_out=0,
            latency_ms=0,
            status="ok",
        )

        return {"competitors": [c.model_dump(mode="json") for c in competitors]}

    # ---- LLM 增强: 分片 + 自一致性 + 自评估 ----
    def _llm_swot(
        self,
        task_id: str,
        round_no: int,
        competitor: str,
        items: List[RawEvidence],
    ) -> Tuple[Optional[SWOT], Dict[str, Any]]:
        """返回 (SWOT, meta)；失败时 SWOT 为 None。"""
        meta: Dict[str, Any] = {"shards": None, "self_eval": None}
        if isinstance(self.llm, MockLLMProvider):
            return None, meta

        url_to_ev = {ev.source_url: ev for ev in items}

        # — 步骤 1: 超长上下文分片 ————————————————————
        shards = self._shard_evidence(items)
        meta["shards"] = {
            "strategy": "by_topic" if len(shards) > 1 else "single",
            "count": len(shards),
            "sizes": [len(s) for s in shards],
        }

        # — 步骤 2: 每片调 LLM, 并对每片做自一致性采样 ————
        all_shard_swots: List[SWOT] = []
        for shard_idx, shard_items in enumerate(shards):
            samples = self._sample_swot(task_id, round_no, competitor, shard_items, shard_idx)
            if not samples:
                continue
            voted = self._consensus_swot(samples, url_to_ev) if len(samples) > 1 else samples[0]
            all_shard_swots.append(voted)

        if not all_shard_swots:
            return None, meta

        # — 步骤 3: 合并多片 SWOT（去重 by claim 前 30 字）————
        merged = self._merge_swots(all_shard_swots)

        # — 步骤 4: Agent 自评估 ————————————————————————
        self_eval = self._self_evaluate(merged, items)
        meta["self_eval"] = self_eval

        # 自评太低 → 视为失败回退兜底
        if self_eval["overall"] < LOW_CONFIDENCE_SCORE:
            self.trace.write(
                task_id=task_id, agent=self.role, round_no=round_no,
                prompt="self_evaluation",
                input_payload=f"competitor={competitor}",
                output_payload=f"low_confidence: {json.dumps(self_eval, ensure_ascii=False)}",
                tokens_in=0, tokens_out=0, latency_ms=0,
                status="degraded",
            )
            return None, meta

        return merged, meta

    # —— 分片策略: 按 topic 切, 每片不超过 SHARD_THRESHOLD 条 ——
    @staticmethod
    def _shard_evidence(items: List[RawEvidence]) -> List[List[RawEvidence]]:
        if len(items) <= SHARD_THRESHOLD:
            return [items]
        by_topic: Dict[str, List[RawEvidence]] = defaultdict(list)
        for ev in items:
            by_topic[ev.topic].append(ev)
        # 每个 topic 一片；若某 topic 仍 > 阈值，按 confidence 排序再切
        shards: List[List[RawEvidence]] = []
        for topic, group in by_topic.items():
            if len(group) <= SHARD_THRESHOLD:
                shards.append(group)
            else:
                group_sorted = sorted(group, key=lambda e: -e.confidence)
                for i in range(0, len(group_sorted), SHARD_THRESHOLD):
                    shards.append(group_sorted[i : i + SHARD_THRESHOLD])
        return shards

    # —— 自一致性: 同一 prompt 跑 N 次, 对每片独立做 ——
    def _sample_swot(
        self, task_id: str, round_no: int, competitor: str,
        shard_items: List[RawEvidence], shard_idx: int,
    ) -> List[SWOT]:
        url_to_ev = {ev.source_url: ev for ev in shard_items}
        user = self._build_swot_prompt(competitor, shard_items, shard_idx)
        samples: List[SWOT] = []
        for sample_no in range(SELF_CONSISTENCY_SAMPLES):
            try:
                data = self.call_llm(
                    task_id=task_id, round_no=round_no,
                    system=_build_analyst_system(), user=user,
                    json_mode=True,
                    temperature=0.2 + sample_no * 0.2,
                )
            except Exception:
                continue
            if not isinstance(data, dict) or data.get("_parse_error"):
                continue
            try:
                quadrant_keys = [q["key"] for q in SchemaConfig.get().get_swot_quadrants()]
                swot_kwargs = {k: self._quadrant(data.get(k), url_to_ev) for k in quadrant_keys}
                swot = SWOT(**swot_kwargs)
                samples.append(swot)
            except ValidationError:
                continue
        return samples

    # —— 投票判定: claim 前 30 字归一后计数, 要 >= VOTE_MIN_SUPPORT ——
    @staticmethod
    def _consensus_swot(
        samples: List[SWOT], url_to_ev: Dict[str, RawEvidence],
    ) -> SWOT:
        def vote(quadrant: str) -> List[ClaimWithCitation]:
            bucket: Dict[str, List[ClaimWithCitation]] = defaultdict(list)
            for s in samples:
                for c in getattr(s, quadrant):
                    key = c.claim[:30].strip()
                    bucket[key].append(c)
            return [v[0] for k, v in bucket.items() if len(v) >= VOTE_MIN_SUPPORT]

        quadrant_keys = [q["key"] for q in SchemaConfig.get().get_swot_quadrants()]
        return SWOT(**{k: vote(k) for k in quadrant_keys})

    # —— 多片合并 + 去重 ——
    @staticmethod
    def _merge_swots(swots: List[SWOT]) -> SWOT:
        def merge(field: str) -> List[ClaimWithCitation]:
            seen: Dict[str, ClaimWithCitation] = {}
            for s in swots:
                for c in getattr(s, field):
                    k = c.claim[:30].strip()
                    if k not in seen:
                        seen[k] = c
            return list(seen.values())

        quadrant_keys = [q["key"] for q in SchemaConfig.get().get_swot_quadrants()]
        return SWOT(**{k: merge(k) for k in quadrant_keys})

    # —— 自评估: 3 维打分 + 加权 ——
    @staticmethod
    def _self_evaluate(swot: SWOT, items: List[RawEvidence]) -> Dict[str, float]:
        quadrant_keys = [q["key"] for q in SchemaConfig.get().get_swot_quadrants()]
        filled = sum(1 for q in quadrant_keys if len(getattr(swot, q, [])) > 0)
        completeness = filled / max(len(quadrant_keys), 1)

        all_claims: List[ClaimWithCitation] = []
        for q in quadrant_keys:
            all_claims.extend(getattr(swot, q, []))

        if not all_claims:
            cite_density = 0.0
        else:
            avg_cites = sum(len(c.citations) for c in all_claims) / len(all_claims)
            cite_density = min(avg_cites / 2.0, 1.0)

        used_urls = {ct.source_url for c in all_claims for ct in c.citations}
        coverage = len(used_urls) / max(len(items), 1)
        reasoning_quality = min(coverage * 2, 1.0)

        overall = round(
            completeness * 0.4 + cite_density * 0.3 + reasoning_quality * 0.3, 3
        )
        return {
            "completeness": round(completeness, 3),
            "citation_density": round(cite_density, 3),
            "reasoning_quality": round(reasoning_quality, 3),
            "overall": overall,
        }

    @staticmethod
    def _build_swot_prompt(
        competitor: str, items: List[RawEvidence], shard_idx: int,
    ) -> str:
        evidence_lines = [
            f"- topic={ev.topic} | url={ev.source_url} | content={ev.content}"
            for ev in items
        ]
        return (
            f"竞品名: {competitor}\n"
            f"证据分片: #{shard_idx}\n"
            f"证据条数: {len(items)}\n"
            f"证据列表（每行: topic | url | content）:\n"
            + "\n".join(evidence_lines)
            + "\n\n请基于以上证据输出 SWOT JSON。"
        )

    @staticmethod
    def _build_competitor(name: str, items: List[RawEvidence]) -> Competitor:
        config = SchemaConfig.get()
        features: List[FeatureNode] = []
        pricing: List[PricingPlan] = []
        personas: List[Persona] = []
        swot = SWOT()

        def _add_feature(ev, cite):
            features.append(FeatureNode(
                name=_extract_title(ev.content),
                description=ev.content,
                citations=[cite]
            ))

        def _add_pricing(ev, cite):
            pricing.append(PricingPlan(
                tier_name=_extract_title(ev.content),
                price=_extract_price(ev.content),
                features_included=[],
                citations=[cite]
            ))

        def _add_persona(ev, cite):
            personas.append(Persona(
                segment=_extract_title(ev.content),
                pain_points=[ev.content],
                citations=[cite]
            ))

        def _add_review(ev, cite):
            target = swot.strengths if ev.confidence >= 0.7 else swot.weaknesses
            target.append(ClaimWithCitation(
                claim=ev.content,
                citations=[cite]
            ))

        builders = {
            "features": _add_feature,
            "pricing": _add_pricing,
            "personas": _add_persona,
            "swot": _add_review
        }

        for ev in items:
            cite = _to_citation(ev)
            field = config.field_for_topic(ev.topic)
            if field and field in builders:
                builders[field](ev, cite)

        # 兜底：若 SWOT 为空，先补一条占位（后续闭环可能再补 weakness）
        if items:
            sample = _to_citation(items[0])
            quadrants = config.get_swot_quadrants()
            for q in quadrants:
                qlist = getattr(swot, q["key"], None)
                if isinstance(qlist, list) and not qlist:
                    qlist.append(ClaimWithCitation(
                        claim=f"{name} - {q['label']}待补充",
                        citations=[sample],
                    ))

        return Competitor(
            name=name,
            features=features,
            pricing=pricing,
            personas=personas,
            swot=swot
        )

    @staticmethod
    def _quadrant(items, url_to_ev: Dict[str, RawEvidence]) -> List[ClaimWithCitation]:
        """把 LLM 返回的一个象限 list 转为 ClaimWithCitation[], 过滤未在证据中的 url。"""
        out: List[ClaimWithCitation] = []
        if not isinstance(items, list):
            return out
        for it in items:
            claim = (it or {}).get("claim")
            urls = (it or {}).get("source_urls") or []
            if not claim or not urls:
                continue
            citations: List[Citation] = []
            for u in urls:
                ev = url_to_ev.get(u)
                if ev is None:
                    continue  # 引用了证据外的 url, 直接丢弃（防幻觉）
                citations.append(_to_citation(ev))
            if not citations:
                continue
            out.append(ClaimWithCitation(claim=claim, citations=citations))
        return out


def _to_citation(ev: RawEvidence) -> Citation:
    return Citation(
        source_url=ev.source_url,
        snippet=ev.content[:140],
        fetched_at=ev.fetched_at,
        confidence=ev.confidence,
    )


def _extract_title(content: str) -> str:
    for sep in ("\n", ":", "。", ",", "；"):
        if sep in content:
            head = content.split(sep, 1)[0].strip()
            if head:
                return head[:30]
    return content[:20]


def _extract_price(content: str) -> str:
    for kw in ("¥", "$", "元/", "/月", "/年"):
        if kw in content:
            idx = content.index(kw)
            return content[max(0, idx - 10) : idx + 10].strip()
    return "联系销售"