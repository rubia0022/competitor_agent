"""
DAG 可视化:
- 静态图: 从 LangGraph 编译产物导出系统拓扑 (Mermaid + ASCII)
- 动态回放: 根据某次 task 的 messages，绘制"实际走过的路径"
把闭环 (reject → revise) 作为带颜色/编号的边画出来
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List


# ---- 静态图 ----
def export_static_mermaid(compiled_graph) -> str:
    """直接调用 LangGraph 自带的 mermaid 导出。"""
    return compiled_graph.get_graph().draw_mermaid()


def export_static_ascii() -> str:
    """ASCII 拓扑，方便在终端展示。"""
    return r"""
    ┌─────────┐     ┌──────────┐     ┌─────────────┐
    │ collect │────>│ analyze  │────>│ qa_analyst   │
    └─────────┘     └──────────┘     └─────────────┘
         ↑             │ reject
         │打回(high)   │
         └─────────────┘
                      │ approve
                      ▼
                    ┌────────┐     ┌────────────┐
                    │ write  │────>│ qa_writer   │────> approve ──> END
                    └────────┘     └────────────┘
                                      │ reject (≤3 轮)
                                      ▼
                                  (重写 write)
    """


# ---- 动态回放 ----
NODE_OF = {
    "collector": "collect",
    "analyst": "analyze",
    "writer": "write",
    "qa": "qa",
}


def replay_to_mermaid(messages: List[Dict[str, Any]]) -> str:
    """根据消息链生成"本次实际执行路径"的 mermaid sequenceDiagram。
    每条边带 (轮次, intent)，reject/approve 着色，便于看出闭环。
    """
    lines = ["sequenceDiagram", "    autonumber"]
    actors = []
    for role in ("collector", "analyst", "writer", "qa"):
        if any(m["from_agent"] == role or m["to_agent"] == role for m in messages):
            actors.append(role)
            lines.append(f"    participant {role}")
    for m in messages:
        arrow = "->>"  # 默认实线
        note = ""
        if m["intent"] == "reject":
            arrow = "-x"
            note = "❌"
        elif m["intent"] == "approve":
            note = "✅"
        elif m["intent"] == "revise":
            note = "🔁"
        lines.append(
            f"    {m['from_agent']}{arrow}{m['to_agent']}: r{m['round_no']} {m['intent']}{note}"
        )
    return "\n".join(lines)


def replay_to_ascii(messages: List[Dict[str, Any]]) -> str:
    """终端友好的回放视图：按时间序列列出全部跳转，闭环用 ↻ 标记。"""
    lines = []
    last_round = -1
    for m in messages:
        if m["round_no"] != last_round:
            sep = "-" * 60
            lines.append(f"\n — round {m['round_no']} {sep[:50]}")
            last_round = m["round_no"]
        icon = {
            "produce": "✨",
            "revise": "🔁",
            "reject": "❌",
            "approve": "✅",
            "review": "🔍",
        }.get(m["intent"], " ")
        lines.append(
            f"  {icon}  {m['from_agent']:>10s} ──> {m['to_agent']:<10s}  ({m['intent']})"
        )
    return "\n".join(lines)


def replay_stats(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总：每个 Agent 被调用次数、每个 intent 的出现次数、闭环触发次数。"""
    by_agent = defaultdict(int)
    by_intent = defaultdict(int)
    rounds = set()
    for m in messages:
        by_agent[m["from_agent"]] += 1
        by_intent[m["intent"]] += 1
        rounds.add(m["round_no"])
    return {
        "total_messages": len(messages),
        "rounds": sorted(rounds),
        "loop_triggered": by_intent.get("reject", 0),
        "by_agent": dict(by_agent),
        "by_intent": dict(by_intent),
    }