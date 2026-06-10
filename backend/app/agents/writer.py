"""
Writer Agent: 把 Competitor[] 渲染成 Markdown 报告。
边界: 禁止引入 Analyst 输出之外的新事实。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from app.agents.base import BaseAgent
from app.config.schema_config import SchemaConfig
from app.schemas import Competitor, CompetitorReport


WRITER_SYSTEM = """[WRITER]
你是企业竞品分析"报告撰写"角色。职责:
1. 仅根据传入的 Competitor 结构化数据生成 Markdown 报告；
2. 禁止引入新的事实/数据/竞品；
3. 报告每条结论后必须用 [n] 形式标注引用编号。
"""


class WriterAgent(BaseAgent):
    role = "writer"

    def run(
        self,
        task_id: str,
        round_no: int,
        *,
        industry: str,
        competitors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        comp_objs = [Competitor(**c) for c in competitors]
        md, ref_count = self._render(industry, comp_objs)

        report = CompetitorReport(
            title=f"{industry} 行业竞品分析报告",
            industry=industry,
            competitors=comp_objs,
            markdown=md,
            generated_at=datetime.utcnow(),
        )

        self.trace.write(
            task_id=task_id,
            agent=self.role,
            round_no=round_no,
            prompt=WRITER_SYSTEM,
            input_payload=f"competitors={[c.name for c in comp_objs]}",
            output_payload=f"markdown_len={len(md)}, refs={ref_count}",
            tokens_in=0,
            tokens_out=0,
            latency_ms=0,
            status="ok",
        )

        return {"report": report.model_dump(mode="json")}

    @staticmethod
    def _render(industry: str, competitors: List[Competitor]) -> tuple[str, int]:
        config = SchemaConfig.get()
        lines: List[str] = []
        refs: List[str] = []

        def cite(urls: List[str]) -> str:
            ids = []
            for u in urls:
                if u not in refs:
                    refs.append(u)
                ids.append(str(refs.index(u) + 1))
            return "[" + ",".join(ids) + "]"

        def _render_features(c: Competitor) -> None:
            if c.features:
                lines.append("\n### 核心功能\n")
                for f in c.features:
                    urls = [ct.source_url for ct in f.citations]
                    lines.append(f"- **{f.name}**: {f.description} {cite(urls)}")

        def _render_pricing(c: Competitor) -> None:
            if c.pricing:
                lines.append("\n### 定价\n")
                for p in c.pricing:
                    urls = [ct.source_url for ct in p.citations]
                    lines.append(f"- {p.tier_name} – {p.price} {cite(urls)}")

        def _render_personas(c: Competitor) -> None:
            if c.personas:
                lines.append("\n### 用户画像\n")
                for pe in c.personas:
                    urls = [ct.source_url for ct in pe.citations]
                    pains = "; ".join(pe.pain_points)
                    lines.append(f"- {pe.segment}: {pains} {cite(urls)}")

        def _render_swot(c: Competitor) -> None:
            lines.append("\n### SWOT\n")
            for q in config.get_swot_quadrants():
                items = getattr(c.swot, q["key"], [])
                if items:
                    lines.append(f"**{q['label']}**")
                    for it in items:
                        urls = [ct.source_url for ct in it.citations]
                        lines.append(f"- {it.claim} {cite(urls)}")

        section_renderers = {
            "features": _render_features,
            "pricing": _render_pricing,
            "personas": _render_personas,
            "swot": _render_swot,
        }

        lines.append(f"# {industry} 行业竞品分析报告\n")
        lines.append(f"生成时间: {datetime.utcnow().isoformat()[:19]}\n")
        lines.append("## 1. 概览\n")
        lines.append(f"本报告覆盖 {len(competitors)} 个竞品: " + "、".join(c.name for c in competitors) + "\n")

        for idx, c in enumerate(competitors, 1):
            lines.append(f"\n## {idx}. {c.name}\n")
            for section in config.get_writer_sections():
                renderer = section_renderers.get(section)
                if renderer:
                    renderer(c)

        if refs:
            lines.append("\n## 参考来源\n")
            for i, u in enumerate(refs, 1):
                lines.append(f"[{i}] {u}")

        return "\n".join(lines), len(refs)