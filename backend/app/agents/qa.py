"""
QA Agent: 对 Analyst / Writer 产物做事实+Schema 双重质检，输出 QAReport。

校验维度:
1. Schema 严格校验 (Pydantic)
2. 每条结论的 Citation 是否为空
3. SWOT 四象限是否至少各有 1 条
4. 功能/定价/画像至少要有
5. 报告 markdown 是否包含所有引用
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import ValidationError

from app.agents.base import BaseAgent
from app.config.schema_config import SchemaConfig
from app.schemas import (
    Competitor,
    CompetitorReport,
    Issue,
    QAReport,
)


QA_SYSTEM = """[QA]
你是企业竞品分析"质检"角色。职责:
1. 严格按 Schema 校验上游产物；
2. 检查每条结论是否带 Citation；
3. 对不足以高 severity 标注，能定位到 field_path；
4. 输出 QAReport JSON，禁止自行修复。
"""


class QAAgent(BaseAgent):
    role = "qa"

    def run_for_analyst(
        self,
        task_id: str,
        round_no: int,
        *,
        competitors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        issues: List[Issue] = []

        if not competitors:
            issues.append(
                Issue(field_path="competitors", reason="Analyst 产物为空", severity="high")
            )

        for c_idx, raw_c in enumerate(competitors):
            path = f"competitors[{raw_c.get('name', c_idx)}]"
            # Schema 校验
            try:
                c = Competitor(**raw_c)
            except ValidationError as e:
                issues.append(
                    Issue(
                        field_path=path,
                        reason=f"Schema 校验失败: {e.errors()[:2]}",
                        severity="high",
                    )
                )
                continue

            config = SchemaConfig.get()
            for field_cfg in config.get_active_fields():
                fkey = field_cfg["key"]
                if fkey == "swot":
                    continue
                if fkey in config.get_qa_required_fields() and not getattr(c, fkey, None):
                    issues.append(
                        Issue(field_path=f"{path}.{fkey}", reason=f"缺少{field_cfg['label']}", severity="high")
                    )

            for q in config.get_swot_quadrants():
                if len(getattr(c.swot, q["key"], [])) == 0:
                    issues.append(
                        Issue(
                            field_path=f"{path}.swot.{q['key']}",
                            reason=f"SWOT 缺少{q['label']}",
                            severity="mid",
                        )
                    )

        report = QAReport(
            target_agent="analyst",
            is_pass=all(i.severity != "high" for i in issues) and len(issues) <= 2,
            issues=issues,
            summary=f"共发现 {len(issues)} 个问题"
            + ("，需要打回重做" if issues else "，质检通过"),
        )

        self._write_trace(task_id, round_no, target="analyst", issues=len(issues), is_pass=report.is_pass)
        return {"qa_report": report.model_dump(mode="json")}

    def run_for_writer(
        self,
        task_id: str,
        round_no: int,
        *,
        report: Dict[str, Any],
        competitors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        issues: List[Issue] = []
        try:
            rep = CompetitorReport(**report)
        except ValidationError as e:
            issues.append(Issue(field_path="report", reason=f"Report Schema 失败: {e.errors()[:2]}", severity="high"))
            qa = QAReport(target_agent="writer", is_pass=False, issues=issues, summary="报告 Schema 校验失败")
            self._write_trace(task_id, round_no, target="writer", issues=len(issues), is_pass=False)
            return {"qa_report": qa.model_dump(mode="json")}

        # 引用完整性: 每个竞品在 markdown 中必须出现
        for c in competitors:
            if c["name"] not in rep.markdown:
                issues.append(
                    Issue(
                        field_path=f"report.markdown[{c['name']}]",
                        reason="报告未提及该竞品",
                        severity="high",
                    )
                )

        # 引用编号是否出现
        if "[1]" not in rep.markdown:
            issues.append(Issue(field_path="report.markdown", reason="报告缺少引用编号", severity="mid"))

        qa = QAReport(
            target_agent="writer",
            is_pass=all(i.severity != "high" for i in issues),
            issues=issues,
            summary=f"报告质检: {len(issues)} 个问题",
        )

        self._write_trace(task_id, round_no, target="writer", issues=len(issues), is_pass=qa.is_pass)
        return {"qa_report": qa.model_dump(mode="json")}

    def run(self, task_id: str, round_no: int, **kwargs):  # 兼容 BaseAgent 抽象方法
        raise NotImplementedError("Use run_for_analyst / run_for_writer")

    def _write_trace(self, task_id, round_no, *, target: str, issues: int, is_pass: bool):
        self.trace.write(
            task_id=task_id,
            agent=self.role,
            round_no=round_no,
            prompt=QA_SYSTEM,
            input_payload=f"target={target}",
            output_payload=f"issues={issues}, pass={is_pass}",
            tokens_in=0,
            tokens_out=0,
            latency_ms=0,
            status="ok",
        )