"""
CLI 入口:

用法:
python -m app.cli run --industry "企业协同办公" --competitors "飞书,钉钉,企业微信"
python -m app.cli run --llm doubao --industry ... --competitors ...
python -m app.cli trace --task-id <uuid>
python -m app.cli graph
python -m app.cli replay --task-id <uuid>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

from app.agents import AnalystAgent, CollectorAgent, QAAgent, WriterAgent
from app.orchestrator import (
    Orchestrator,
    export_static_ascii,
    export_static_mermaid,
    replay_stats,
    replay_to_ascii,
    replay_to_mermaid,
)
from app.providers import (
    DoubaoLLMProvider,
    LLMProvider,
    MockLLMProvider,
    MockSearchProvider,
)
from app.storage import TraceStore
from app.storage.langfuse_tracer import LangfuseTracer

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
MOCK_DIR = DATA_DIR / "mock"
OUTPUT_DIR = DATA_DIR / "output"
DB_PATH = DATA_DIR / "competitor_agent.sqlite"
ENV_PATH = BACKEND_DIR / ".env"

# 启动时尝试加载 backend/.env（凭据持久化）；缺失则静默跳过
try:
    from dotenv import load_dotenv
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
except ImportError:
    pass


def _build_llm(kind: str) -> LLMProvider:
    """根据开关返回 LLM Provider。"""
    if kind == "mock":
        return MockLLMProvider()
    if kind == "doubao":
        api_key = os.environ.get("DOUBAO_API_KEY")
        endpoint = os.environ.get("DOUBAO_ENDPOINT")
        if not api_key or not endpoint:
            print(
                "[fatal] 缺少环境变量 DOUBAO_API_KEY / DOUBAO_ENDPOINT。\n"
                "  export DOUBAO_API_KEY=ark-xxxx\n"
                "  export DOUBAO_ENDPOINT=ep-xxxx",
                file=sys.stderr,
            )
            sys.exit(2)
        return DoubaoLLMProvider(api_key=api_key, endpoint_id=endpoint)
    raise ValueError(f"未知 LLM 类型: {kind}")


def _build_orchestrator(llm_kind: str = "mock") -> Orchestrator:
    from app.storage.langfuse_tracer import LangfuseTracer
    langfuse = LangfuseTracer()
    trace = TraceStore(DB_PATH, langfuse=langfuse)
    llm = _build_llm(llm_kind)
    search = MockSearchProvider(MOCK_DIR)
    return Orchestrator(
        collector=CollectorAgent(llm, trace, search),
        analyst=AnalystAgent(llm, trace),
        writer=WriterAgent(llm, trace),
        qa=QAAgent(llm, trace),
        trace=trace,
    )


def cmd_run(industry: str, competitors: List[str], llm_kind: str) -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    orch = _build_orchestrator(llm_kind=llm_kind)
    result = orch.run(industry=industry, competitors=competitors)
    task_id = result["task_id"]
    out_dir = OUTPUT_DIR / task_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. JSON 报告
    (out_dir / "report.json").write_text(
        json.dumps(result["report"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 2. Markdown 报告（方便人读）
    if result["report"] and result["report"].get("markdown"):
        (out_dir / "report.md").write_text(result["report"]["markdown"], encoding="utf-8")

    # 3. 摘要
    summary = {
        "task_id": task_id,
        "status": result["status"],
        "rounds": result["rounds"],
        "qa_analyst_pass": (result.get("qa_analyst") or {}).get("is_pass"),
        "qa_writer_pass": (result.get("qa_writer") or {}).get("is_pass"),
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 4. DAG 可视化: 本次实际路径 + 系统拓扑
    messages = orch.trace.list_messages(task_id)
    (out_dir / "replay.mmd").write_text(replay_to_mermaid(messages), encoding="utf-8")
    (out_dir / "graph.mmd").write_text(export_static_mermaid(orch._graph), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 64)
    print(f"报告已生成: {out_dir / 'report.md'}")
    print(f"实际路径图: {out_dir / 'replay.mmd'} (mermaid)")
    print(f"系统拓扑图: {out_dir / 'graph.mmd'} (mermaid)")
    print(f"查看 trace: python -m app.cli trace --task-id {task_id}")
    print(f"动态回放: python -m app.cli replay --task-id {task_id}")
    return 0


def cmd_trace(task_id: str) -> int:
    trace = TraceStore(DB_PATH)
    traces = trace.list_traces(task_id)
    messages = trace.list_messages(task_id)
    print(f"# Task {task_id}")
    print(f"\n## Messages ({len(messages)})\n")
    for m in messages:
        print(f"  r{m['round_no']} [{m['from_agent']:>10s} -> {m['to_agent']:<10s}] {m['intent']:<8s}")
    print(f"\n## Traces ({len(traces)})\n")
    for t in traces:
        out_short = (t["output_payload"] or "")[:80].replace("\n", " ")
        print(f"  r{t['round_no']} [{t['agent']:>12s}] {t['status']:>4s} {t['latency_ms']:>4d}ms  {out_short}")
    return 0


def cmd_graph(fmt: str) -> int:
    """导出系统静态 DAG（不依赖具体 task）。"""
    if fmt == "mermaid":
        orch = _build_orchestrator(llm_kind="mock")
        print(export_static_mermaid(orch._graph))
    else:
        print(export_static_ascii())
    return 0


def cmd_replay(task_id: str, fmt: str) -> int:
    """根据 messages 回放某次 task 的实际执行路径。"""
    trace = TraceStore(DB_PATH)
    messages = trace.list_messages(task_id)
    if not messages:
        print(f"[warn] task_id={task_id} 没有消息记录")
        return 1
    print(f"# Replay  task:{task_id}")
    stats = replay_stats(messages)
    print("\n## Stats")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n## Path ({fmt})\n")
    if fmt == "mermaid":
        print(replay_to_mermaid(messages))
    else:
        print(replay_to_ascii(messages))
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="competitor-agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="跑一次完整竞品分析")
    p_run.add_argument("--industry", required=True, help="行业，如 '企业协同办公'")
    p_run.add_argument("--competitors", required=True, help="逗号分隔的竞品名，如 '飞书,钉钉,企业微信'")
    p_run.add_argument("--llm", choices=["mock", "doubao"], default="doubao",
                      help="LLM 提供方。doubao 需设置 DOUBAO_API_KEY / DOUBAO_ENDPOINT；--llm mock 走纯规则演示")

    p_tr = sub.add_parser("trace", help="查看一个 task 的 trace")
    p_tr.add_argument("--task-id", required=True)

    p_g = sub.add_parser("graph", help="导出系统静态 DAG")
    p_g.add_argument("--format", choices=["mermaid", "ascii"], default="ascii")

    p_rp = sub.add_parser("replay", help="回放某次 task 的实际执行路径")
    p_rp.add_argument("--task-id", required=True)
    p_rp.add_argument("--format", choices=["mermaid", "ascii"], default="ascii")

    args = parser.parse_args(argv)
    if args.cmd == "run":
        competitors = [c.strip() for c in args.competitors.split(",") if c.strip()]
        return cmd_run(args.industry, competitors, args.llm)
    elif args.cmd == "trace":
        return cmd_trace(args.task_id)
    elif args.cmd == "graph":
        return cmd_graph(args.format)
    elif args.cmd == "replay":
        return cmd_replay(args.task_id, args.format)
    return 1


if __name__ == "__main__":
    sys.exit(main())