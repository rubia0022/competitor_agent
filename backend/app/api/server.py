"""
FastAPI 应用：暴露任务创建、查询、SSE 实时进度。

启动:
    cd backend && . .venv/bin/python -m app.api.server

访问:
    POST /api/tasks              创建任务（异步）
    GET  /api/tasks              列出最近任务
    GET  /api/tasks/{id}         任务摘要 + 报告
    GET  /api/tasks/{id}/trace   trace 列表
    GET  /api/tasks/{id}/messages 消息列表
    GET  /api/tasks/{id}/replay  回放视图（mermaid + ascii + stats）
    GET  /api/tasks/{id}/stream  SSE 实时推送 trace 增量
    GET  /api/graph              系统静态 DAG（mermaid）
    GET  /                       前端静态页（如存在）
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.agents import AnalystAgent, CollectorAgent, QAAgent, WriterAgent
from app.orchestrator import (
    Orchestrator,
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

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BACKEND_DIR / "data"
MOCK_DIR = DATA_DIR / "mock"
OUTPUT_DIR = DATA_DIR / "output"
DB_PATH = DATA_DIR / "competitor_agent.sqlite"
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"

# .env 加载（与 CLI 一致）
ENV_PATH = BACKEND_DIR / ".env"
try:
    from dotenv import load_dotenv
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
except ImportError:
    pass

import os


def _build_llm(kind: str) -> LLMProvider:
    if kind == "mock":
        return MockLLMProvider()
    if kind == "doubao":
        api_key = os.environ.get("DOUBAO_API_KEY")
        endpoint = os.environ.get("DOUBAO_ENDPOINT")
        if not api_key or not endpoint:
            raise HTTPException(
                status_code=500,
                detail="缺少 DOUBAO_API_KEY / DOUBAO_ENDPOINT（请检查 backend/.env）",
            )
        return DoubaoLLMProvider(api_key=api_key, endpoint_id=endpoint)
    raise HTTPException(status_code=400, detail=f"未知 LLM 类型: {kind}")


def _build_orchestrator(llm_kind: str) -> Orchestrator:
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


# —— Request / Response Schemas ——
class CreateTaskRequest(BaseModel):
    industry: str = Field(..., examples=["企业协同办公"])
    competitors: List[str] = Field(..., examples=[["飞书", "钉钉", "企业微信"]])
    llm: str = Field("doubao", description="mock | doubao")


class TaskSummary(BaseModel):
    task_id: str
    industry: str
    competitors: List[str]
    status: str
    rounds: int
    created_at: Optional[str]
    finished_at: Optional[str]


class ResumeRequest(BaseModel):
    competitors: Optional[List[Dict[str, Any]]] = None
    report: Optional[Dict[str, Any]] = None
    llm: str = Field("mock", description="mock | doubao")


# —— 异步任务执行（后台线程跑 orchestrator）——
def _run_task_async(task_id_holder: dict, industry: str, competitors: List[str], llm_kind: str) -> None:
    """后台线程入口。orchestrator 会自己创建一个 task_id，我们事先无法预测。
    这里改成捕获 orchestrator.run 返回的 task_id 写回 holder。"""
    try:
        orch = _build_orchestrator(llm_kind=llm_kind)
        result = orch.run(industry=industry, competitors=competitors)
        task_id_holder["task_id"] = result["task_id"]
        task_id_holder["status"] = result["status"]
        # 落盘报告
        out_dir = OUTPUT_DIR / result["task_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        if result.get("report"):
            (out_dir / "report.json").write_text(
                json.dumps(result["report"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if result["report"].get("markdown"):
            (out_dir / "report.md").write_text(result["report"]["markdown"], encoding="utf-8")
    except Exception as e:
        task_id_holder["error"] = f"{type(e).__name__}: {e}"
        task_id_holder["status"] = "failed"


# —— 全局：进行中的任务（thread + holder）——
_inflight: Dict[str, Dict[str, Any]] = {}  # session_id -> {thread, holder}


# —— App ——
app = FastAPI(
    title="AI 竞品分析 Agent 协作系统",
    description="多 Agent 协作的竞品分析系统，覆盖采集→编排→存储→接口→前端全链路。",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat()}


@app.post("/api/tasks", status_code=202)
def create_task(req: CreateTaskRequest):
    """创建任务（异步）。返回 session_id，前端用它轮询 status；任务真正落盘后会有 task_id。"""
    session_id = f"sess-{int(time.time() * 1000)}"
    holder: Dict[str, Any] = {
        "session_id": session_id,
        "task_id": None,
        "status": "running",
        "started_at": datetime.utcnow().isoformat(),
    }
    t = threading.Thread(
        target=_run_task_async,
        args=(holder, req.industry, req.competitors, req.llm),
        daemon=True,
    )
    t.start()
    _inflight[session_id] = {"thread": t, "holder": holder}
    return {"session_id": session_id, "status": "running"}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    """查询某个 session 的进度（任务真正 task_id 一旦生成就能拿到）。"""
    item = _inflight.get(session_id)
    if not item:
        raise HTTPException(404, "session not found")
    return item["holder"]


@app.get("/api/tasks")
def list_tasks(limit: int = 20):
    """最近的任务列表（直接从 SQLite 读）。"""
    trace = TraceStore(DB_PATH)
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT task_id, industry, competitors_json, status, rounds, created_at, finished_at"
        " FROM tasks ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    out = []
    for r in rows:
        out.append({
            "task_id": r["task_id"],
            "industry": r["industry"],
            "competitors": json.loads(r["competitors_json"] or "[]"),
            "status": r["status"],
            "rounds": r["rounds"],
            "created_at": r["created_at"],
            "finished_at": r["finished_at"],
        })
    return out


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    trace = TraceStore(DB_PATH)
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
    if not row:
        raise HTTPException(404, "task not found")
    report_path = OUTPUT_DIR / task_id / "report.json"
    report = None
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "task_id": row["task_id"],
        "industry": row["industry"],
        "competitors": json.loads(row["competitors_json"] or "[]"),
        "status": row["status"],
        "rounds": row["rounds"],
        "created_at": row["created_at"],
        "finished_at": row["finished_at"],
        "report": report,
    }


@app.get("/api/tasks/{task_id}/trace")
def get_trace(task_id: str):
    trace = TraceStore(DB_PATH)
    return trace.list_traces(task_id)


@app.get("/api/tasks/{task_id}/messages")
def get_messages(task_id: str):
    trace = TraceStore(DB_PATH)
    return trace.list_messages(task_id)


@app.get("/api/tasks/{task_id}/replay")
def get_replay(task_id: str):
    trace = TraceStore(DB_PATH)
    messages = trace.list_messages(task_id)
    if not messages:
        raise HTTPException(404, "no messages for this task")
    return {
        "stats": replay_stats(messages),
        "mermaid": replay_to_mermaid(messages),
        "ascii": replay_to_ascii(messages),
    }


@app.get("/api/tasks/{task_id}/stream")
async def stream_trace(task_id: str):
    """SSE：每 800ms 推送 trace 增量。前端用 EventSource 监听。"""
    async def event_gen():
        trace = TraceStore(DB_PATH)
        seen = 0
        idle = 0
        for _ in range(150):  # 最多 120 秒
            traces = trace.list_traces(task_id)
            new = traces[seen:]
            if new:
                idle = 0
                for t in new:
                    yield {"event": "trace", "data": json.dumps(t, ensure_ascii=False, default=str)}
                seen = len(traces)
            else:
                idle += 1
            # 任务完成判定：查 tasks 表
            import sqlite3
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT status FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if row and row["status"] in ("done", "needs_human", "failed"):
                yield {"event": "done", "data": json.dumps({"status": row["status"]})}
                break
            if idle > 30:  # 24 秒无更新
                yield {"event": "timeout", "data": "{}"}
                break
            await asyncio.sleep(0.8)
    return EventSourceResponse(event_gen())


@app.get("/api/graph")
def get_static_graph():
    """系统静态 DAG（不依赖具体 task）。"""
    orch = _build_orchestrator(llm_kind="mock")  # 仅取 graph 结构
    return {"mermaid": export_static_mermaid(orch._graph)}


@app.get("/api/schema")
def get_schema():
    """返回当前 Schema 配置。"""
    from app.config.schema_config import SchemaConfig
    return SchemaConfig.get().to_dict()


@app.get("/api/metrics")
def get_metrics():
    """运营指标仪表盘：聚合全局关键指标。"""
    import sqlite3 as _sql
    conn = _sql.connect(str(DB_PATH))
    conn.row_factory = _sql.Row

    # 任务维度
    total_tasks = conn.execute("SELECT COUNT(*) AS c FROM tasks").fetchone()["c"]
    done_count = conn.execute("SELECT COUNT(*) AS c FROM tasks WHERE status='done'").fetchone()["c"]
    human_count = conn.execute("SELECT COUNT(*) AS c FROM tasks WHERE status='needs_human'").fetchone()["c"]
    avg_rounds = conn.execute("SELECT AVG(rounds) AS a FROM tasks").fetchone()["a"] or 0

    # 闭环触发率
    tasks_with_reject = conn.execute(
        "SELECT COUNT(DISTINCT task_id) AS c FROM messages WHERE intent='reject'"
    ).fetchone()["c"]
    loop_trigger_rate = tasks_with_reject / max(total_tasks, 1)

    # Token & 延迟
    token_row = conn.execute("SELECT SUM(tokens_in + tokens_out) AS t FROM traces").fetchone()
    total_tokens = token_row["t"] or 0
    latency_row = conn.execute("SELECT AVG(latency_ms) AS a FROM traces WHERE latency_ms > 0").fetchone()
    avg_latency_ms = round(latency_row["a"] or 0)

    # Agent 调用分布
    agent_rows = conn.execute("SELECT agent, COUNT(*) AS c FROM traces GROUP BY agent").fetchall()
    agent_distribution = {r["agent"]: r["c"] for r in agent_rows}

    # 平均自评估分（从 analyst trace 的 output_payload 中解析）
    eval_rows = conn.execute(
        "SELECT output_payload FROM traces WHERE agent='analyst' AND output_payload LIKE '%self_eval%'"
    ).fetchall()
    eval_scores = []
    for r in eval_rows:
        try:
            payload = r["output_payload"]
            if "overall" in payload:
                import re
                m = re.search(r'"overall":\s*([\d\.]+)', payload)
                if m:
                    eval_scores.append(float(m.group(1)))
        except Exception:
            pass
    avg_self_eval = round(sum(eval_scores) / max(len(eval_scores), 1), 3) if eval_scores else None

    # 引用覆盖率（improvement_check 的 coverage 平均值）
    cov_rows = conn.execute(
        "SELECT output_payload FROM traces WHERE prompt='improvement_check'"
    ).fetchall()
    coverages = []
    for r in cov_rows:
        try:
            d = json.loads(r["output_payload"])
            if "coverage" in d:
                coverages.append(d["coverage"])
        except Exception:
            pass
    avg_improvement_coverage = round(sum(coverages) / max(len(coverages), 1), 3) if coverages else None

    return {
        "total_tasks": total_tasks,
        "done_rate": round(done_count / max(total_tasks, 1), 3),
        "needs_human_rate": round(human_count / max(total_tasks, 1), 3),
        "avg_rounds": round(avg_rounds, 2),
        "loop_trigger_rate": round(loop_trigger_rate, 3),
        "total_tokens": total_tokens,
        "avg_latency_ms": avg_latency_ms,
        "agent_distribution": agent_distribution,
        "avg_self_eval": avg_self_eval,
        "avg_improvement_coverage": avg_improvement_coverage,
    }


@app.put("/api/schema")
def update_schema(body: dict):
    """更新 Schema 配置（运行时生效，持久化到 JSON 文件）。"""
    from app.config.schema_config import SchemaConfig
    SchemaConfig.get().update(body)
    return {"ok": True}


@app.get("/api/tasks/{task_id}/state")
def get_task_state(task_id: str):
    """获取 needs_human 任务的保存状态（用于编辑面板）。"""
    trace = TraceStore(DB_PATH)
    saved = trace.load_state(task_id)
    if not saved:
        raise HTTPException(404, "该任务无保存状态")
    return {
        "task_id": task_id,
        "failed_node": saved["failed_node"],
        "state": json.loads(saved["state_json"]),
    }


@app.patch("/api/tasks/{task_id}/state")
def patch_task_state(task_id: str, body: dict):
    """用户编辑后保存状态。"""
    trace = TraceStore(DB_PATH)
    trace.update_state(task_id, body)
    return {"ok": True}


@app.post("/api/tasks/{task_id}/resume")
def resume_task(task_id: str, req: ResumeRequest):
    """恢复 needs_human 任务：应用编辑 + 重跑 QA。"""
    orch = _build_orchestrator(llm_kind=req.llm)
    patches: Dict[str, Any] = {}
    if req.competitors is not None:
        patches["competitors"] = req.competitors
    if req.report is not None:
        patches["report"] = req.report
    result = orch.resume(task_id, patches)
    if result.get("report"):
        out_dir = OUTPUT_DIR / task_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text(
            json.dumps(result["report"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if isinstance(result["report"], dict) and result["report"].get("markdown"):
            (out_dir / "report.md").write_text(
                result["report"]["markdown"], encoding="utf-8"
            )
    return result


# —— 前端静态文件挂载（如存在）——
if FRONTEND_DIR.exists():
    # 默认走 index.html
    @app.get("/")
    def root():
        index = FRONTEND_DIR / "index.html"
        if index.exists():
            return FileResponse(index)
        return JSONResponse(
            {"info": "frontend not built yet. Visit /docs for API."}
        )

    # 把 frontend/ 整体挂在 /static
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
else:
    @app.get("/")
    def root():
        return JSONResponse(
            {
                "info": "Backend ready. Frontend missing. Visit /docs for API.",
                "expected_frontend_dir": str(FRONTEND_DIR),
            }
        )


def main():
    import uvicorn
    uvicorn.run("app.api.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()