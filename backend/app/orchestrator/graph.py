"""
LangGraph StateGraph 编排 DAG + 质检打回闭环 + 改善判定。

节点链:
    collect → analyze → qa_analyst → write → qa_writer → END
              ↑         |           |
              |         ↓           ↓
              |      (打回)      (打回)
              ←---------←---------
条件边: 打回 ≤3 轮

设计要点:
1. 所有跨 Agent 消息走 AgentMessage 落库
2. QA 打回时根据 issue 路由到 collector 或 analyst
3. 重做后做“改善判定”，覆盖 issue 覆盖率 < 80% 视为伪闭环
"""

from __future__ import annotations

import json
import datetime
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph

from app.agents import AnalystAgent, CollectorAgent, QAAgent, WriterAgent
from app.schemas import AgentMessage
from app.storage import TraceStore


MAX_ROUNDS = 3
IMPROVEMENT_THRESHOLD = 0.8


class GraphState(TypedDict, total=False):
    task_id: str
    industry: str
    competitors_input: List[str]
    evidence: List[Dict[str, Any]]
    competitors: List[Dict[str, Any]]
    report: Dict[str, Any]
    qa_analyst: Dict[str, Any]
    qa_writer: Dict[str, Any]
    round_analyst: int
    round_writer: int
    # 闭环改善历史: 保存上一轮的产物 + issue 集，用于改善判定
    prev_competitors: Optional[List[Dict[str, Any]]]
    prev_issues: Optional[List[Dict[str, Any]]]
    needs_human: bool


class Orchestrator:
    def __init__(
        self,
        collector: CollectorAgent,
        analyst: AnalystAgent,
        writer: WriterAgent,
        qa: QAAgent,
        trace: TraceStore,
    ):
        self.collector = collector
        self.analyst = analyst
        self.writer = writer
        self.qa = qa
        self.trace = trace
        self._graph = self._build_graph()

    def _build_graph(self):
        sg = StateGraph(GraphState)
        sg.add_node("collect", self._node_collect)
        sg.add_node("analyze", self._node_analyze)
        sg.add_node("qa_analyst", self._node_qa_analyst)
        sg.add_node("write", self._node_write)
        sg.add_node("qa_writer", self._node_qa_writer)

        sg.set_entry_point("collect")
        sg.add_edge("collect", "analyze")
        sg.add_edge("analyze", "qa_analyst")
        sg.add_conditional_edges(
            "qa_analyst",
            self._route_after_qa_analyst,
            {"write": "write", "collect": "collect", "analyze": "analyze", "end": END},
        )
        sg.add_edge("write", "qa_writer")
        sg.add_conditional_edges(
            "qa_writer",
            self._route_after_qa_writer,
            {"write": "write", "end": END},
        )
        return sg.compile()

    # ---- Nodes ----
    def _node_collect(self, state: GraphState) -> GraphState:
        round_no = state.get("round_analyst", 0)
        issues = (state.get("qa_analyst") or {}).get("issues") if round_no > 0 else None
        out = self.collector.run(
            task_id=state["task_id"],
            round_no=round_no,
            competitors=state["competitors_input"],
            industry=state["industry"],
            revise_issues=issues,
        )
        self._log_message(
            state, from_agent="collector", to_agent="analyst",
            intent="produce" if round_no == 0 else "revise",
            payload={"evidence_count": len(out["evidence"])},
            round_no=round_no,
        )
        return {**state, "evidence": out["evidence"]}

    def _node_analyze(self, state: GraphState) -> GraphState:
        round_no = state.get("round_analyst", 0)
        issues = (state.get("qa_analyst") or {}).get("issues") if round_no > 0 else None
        out = self.analyst.run(
            task_id=state["task_id"],
            round_no=round_no,
            evidence=state["evidence"],
            revise_issues=issues,
        )
        self._log_message(
            state, from_agent="analyst", to_agent="qa",
            intent="produce" if round_no == 0 else "revise",
            payload={"competitors_count": len(out["competitors"])},
            round_no=round_no,
        )
        # 闭环改善判定（仅当不是首轮时执行）
        improvement_note = None
        if round_no > 0 and state.get("prev_competitors") and state.get("prev_issues"):
            improvement_note = self._check_improvement(
                old=state["prev_competitors"],
                new=out["competitors"],
                issues=state["prev_issues"],
            )
        self.trace.write(
            task_id=state["task_id"], agent="orchestrator", round_no=round_no,
            prompt="improvement_check",
            input_payload=f"prev_issues={len(state.get('prev_issues', []))}",
            output_payload=json.dumps(improvement_note, ensure_ascii=False),
            tokens_in=0, tokens_out=0, latency_ms=0,
            status="ok" if improvement_note["pass"] else "degraded",
        )
        return {**state, "competitors": out["competitors"]}

    def _node_qa_analyst(self, state: GraphState) -> GraphState:
        round_no = state.get("round_analyst", 0)
        out = self.qa.run_for_analyst(
            task_id=state["task_id"],
            round_no=round_no,
            competitors=state["competitors"],
        )
        qa = out["qa_report"]
        self._log_message(
            state, from_agent="qa", to_agent="analyst",
            intent="approve" if qa["is_pass"] else "reject",
            payload={"issues": qa["issues"], "summary": qa["summary"]},
            round_no=round_no,
        )
        next_round = round_no if qa["is_pass"] else round_no + 1
        needs_human = (not qa["is_pass"]) and next_round >= MAX_ROUNDS
        new_state = {
            **state,
            "qa_analyst": qa,
            "round_analyst": next_round,
            "needs_human": state.get("needs_human", False) or needs_human,
            "prev_competitors": state.get("competitors"),
            "prev_issues": qa["issues"],
        }
        if needs_human:
            self.trace.save_state(state["task_id"], new_state, "qa_analyst")
        return new_state

    def _node_write(self, state: GraphState) -> GraphState:
        round_no = state.get("round_writer", 0)
        out = self.writer.run(
            task_id=state["task_id"],
            round_no=round_no,
            industry=state["industry"],
            competitors=state["competitors"],
        )
        self._log_message(
            state, from_agent="writer", to_agent="qa",
            intent="produce" if round_no == 0 else "revise",
            payload={"markdown_len": len(out["report"]["markdown"])},
            round_no=round_no,
        )
        return {**state, "report": out["report"]}

    def _node_qa_writer(self, state: GraphState) -> GraphState:
        round_no = state.get("round_writer", 0)
        out = self.qa.run_for_writer(
            task_id=state["task_id"],
            round_no=round_no,
            report=state["report"],
            competitors=state["competitors"],
        )
        qa = out["qa_report"]
        self._log_message(
            state, from_agent="qa", to_agent="writer",
            intent="approve" if qa["is_pass"] else "reject",
            payload={"issues": qa["issues"], "summary": qa["summary"]},
            round_no=round_no,
        )
        next_round = round_no if qa["is_pass"] else round_no + 1
        needs_human = (not qa["is_pass"]) and next_round >= MAX_ROUNDS
        new_state = {
            **state,
            "qa_writer": qa,
            "round_writer": next_round,
            "needs_human": state.get("needs_human", False) or needs_human,
        }
        if needs_human:
            self.trace.save_state(state["task_id"], new_state, "qa_writer")
        return new_state

    # ---- Routing (pure functions, must not mutate state) ----
    def _route_after_qa_analyst(self, state: GraphState) -> str:
        qa = state.get("qa_analyst", {})
        decision, reason = self._decide_after_qa_analyst(state, qa)
        # 决策路径 trace（可观测性硬要求）
        self.trace.write(
            task_id=state["task_id"], agent="orchestrator",
            round_no=state.get("round_analyst", 0),
            prompt="route_after_qa_analyst",
            input_payload=(
                f"is_pass={qa.get('is_pass')}, "
                f"high_issues={sum(1 for i in qa.get('issues', []) if i.get('severity') == 'high')}, "
                f"round={state.get('round_analyst', 0)}, max={MAX_ROUNDS}"
            ),
            output_payload=json.dumps(
                {
                    "decision": decision, "reason": reason,
                    "candidates": ["write", "collect", "analyze", "end"],
                },
                ensure_ascii=False,
            ),
            tokens_in=0, tokens_out=0, latency_ms=0, status="ok",
        )
        return decision

    @staticmethod
    def _decide_after_qa_analyst(state: GraphState, qa: Dict[str, Any]) -> Tuple[str, str]:
        if qa.get("is_pass"):
            return "write", "QA 通过，进入 Writer"
        if state.get("round_analyst", 0) >= MAX_ROUNDS:
            return "end", f"已达最大轮次 {MAX_ROUNDS}，转入人工介入"
        has_high = any(i.get("severity") == "high" for i in qa.get("issues", []))
        if has_high:
            return "collect", "存在 high severity issue（如缺字段），打回 Collector 补采"
        return "analyze", "仅 mid/low severity issue，让 Analyst 重新整合"

    def _route_after_qa_writer(self, state: GraphState) -> str:
        qa = state.get("qa_writer", {})
        decision, reason = self._decide_after_qa_writer(state, qa)
        self.trace.write(
            task_id=state["task_id"], agent="orchestrator",
            round_no=state.get("round_writer", 0),
            prompt="route_after_qa_writer",
            input_payload=(
                f"is_pass={qa.get('is_pass')}, "
                f"round={state.get('round_writer', 0)}, max={MAX_ROUNDS}"
            ),
            output_payload=json.dumps(
                {"decision": decision, "reason": reason, "candidates": ["write", "end"]},
                ensure_ascii=False,
            ),
            tokens_in=0, tokens_out=0, latency_ms=0, status="ok",
        )
        return decision

    @staticmethod
    def _decide_after_qa_writer(state: GraphState, qa: Dict[str, Any]) -> Tuple[str, str]:
        if qa.get("is_pass"):
            return "end", "QA 通过，报告完成"
        if state.get("round_writer", 0) >= MAX_ROUNDS:
            return "end", f"已达最大轮次 {MAX_ROUNDS}，转入人工介入"
        return "write", "报告质检不通过，让 Writer 重写"

    # ---- Helpers ----
    def _log_message(
        self,
        state: GraphState,
        *,
        from_agent: str,
        to_agent: str,
        intent: str,
        payload: Dict[str, Any],
        round_no: int,
    ) -> None:
        msg = AgentMessage(
            task_id=state["task_id"],
            from_agent=from_agent,
            to_agent=to_agent,
            intent=intent,
            payload=payload,
            round_no=round_no,
        )
        self.trace.write_message(msg.model_dump(mode="json"))

    def _check_improvement(
        self,
        *,
        old: List[Dict[str, Any]],
        new: List[Dict[str, Any]],
        issues: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """改善判定：被指出的字段在新版本中是否真的从'空'变'非空'。
        当前实现按 issue.field_path 解析两端 list 长度变化；
        路径形如 'competitors["飞书"].personas' → 找到对应竞品的 personas 长度变化。
        """
        if not issues:
            return {"pass": True, "coverage": 1.0, "note": "no_issues"}
        old_by_name = {c.get("name"): c for c in old or []}
        new_by_name = {c.get("name"): c for c in new or []}
        improved = 0
        for i in issues:
            path = i.get("field_path", "")
            # 解析竞品名
            name = None
            if "[" in path and "]" in path:
                name = path.split("[", 1)[1].split("]", 1)[0]
            # 解析末段字段
            tail = path.split(".")[-1] if "." in path else path
            old_v = (old_by_name.get(name) or {}).get(tail)
            new_v = (new_by_name.get(name) or {}).get(tail)
            old_len = len(old_v) if isinstance(old_v, list) else (1 if old_v else 0)
            new_len = len(new_v) if isinstance(new_v, list) else (1 if new_v else 0)
            if new_len > old_len:
                improved += 1
        coverage = improved / max(len(issues), 1)
        return {
            "pass": coverage >= IMPROVEMENT_THRESHOLD,
            "coverage": round(coverage, 2),
            "note": "improved" if improved > 0 else "no_improvement",
        }

    # ---- Public ----
    def run(self, industry: str, competitors: List[str]) -> Dict[str, Any]:
        task_id = str(uuid4())
        self.trace.create_task(task_id, industry, competitors)
        init_state: GraphState = {
            "task_id": task_id,
            "industry": industry,
            "competitors_input": competitors,
            "round_analyst": 0,
            "round_writer": 0,
            "needs_human": False,
        }
        final: GraphState = self._graph.invoke(init_state)
        status = "needs_human" if final.get("needs_human") else "done"
        self.trace.finish_task(
            task_id,
            status,
            rounds=final.get("round_analyst", 0) + final.get("round_writer", 0),
        )
        return {
            "task_id": task_id,
            "status": status,
            "report": final.get("report"),
            "qa_analyst": final.get("qa_analyst"),
            "qa_writer": final.get("qa_writer"),
            "rounds": {
                "analyst": final.get("round_analyst", 0),
                "writer": final.get("round_writer", 0),
            },
        }

    def resume(self, task_id: str, patches: Dict[str, Any]) -> Dict[str, Any]:
        """从 needs_human 状态恢复：加载保存的 state，应用用户编辑，重跑 QA。"""
        saved = self.trace.load_state(task_id)
        if not saved:
            raise ValueError(f"No saved state for task {task_id}")
        state = json.loads(saved["state_json"])
        failed_node = saved["failed_node"]
        if "competitors" in patches:
            state["competitors"] = patches["competitors"]
        if "report" in patches:
            state["report"] = patches["report"]
        if failed_node == "qa_analyst":
            qa_result = self.qa.run_for_analyst(
                task_id=task_id,
                round_no=state.get("round_analyst", 0),
                competitors=state["competitors"],
            )
            state["qa_analyst"] = qa_result["qa_report"]
            if qa_result["qa_report"]["is_pass"]:
                write_result = self.writer.run(
                    task_id=task_id,
                    round_no=0,
                    industry=state["industry"],
                    competitors=state["competitors"],
                )
                state["report"] = write_result["report"]
                qa_w = self.qa.run_for_writer(
                    task_id=task_id, round_no=0,
                    report=state["report"],
                    competitors=state["competitors"],
                )
                state["qa_writer"] = qa_w["qa_report"]
                status = "done" if qa_w["qa_report"]["is_pass"] else "needs_human"
            else:
                status = "needs_human"
                self.trace.save_state(task_id, state, "qa_analyst")
        elif failed_node == "qa_writer":
            qa_result = self.qa.run_for_writer(
                task_id=task_id,
                round_no=state.get("round_writer", 0),
                report=state["report"],
                competitors=state["competitors"],
            )
            state["qa_writer"] = qa_result["qa_report"]
            status = "done" if qa_result["qa_report"]["is_pass"] else "needs_human"
            if status == "needs_human":
                self.trace.save_state(task_id, state, "qa_writer")
        else:
            raise ValueError(f"Unknown failed_node: {failed_node}")
        rounds = state.get("round_analyst", 0) + state.get("round_writer", 0)
        self.trace.finish_task(task_id, status, rounds)
        return {
            "task_id": task_id,
            "status": status,
            "report": state.get("report"),
            "qa_analyst": state.get("qa_analyst"),
            "qa_writer": state.get("qa_writer"),
        }


def export_static_mermaid(graph) -> str:
    return graph.get_graph().draw_mermaid()