"""
Langfuse 集成：将 Agent 调用链上报到 Langfuse 做外部可观测性。

初始化条件：环境变量 LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY 存在。
否则所有方法 no-op，不影响主流程。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


class LangfuseTracer:
    """Langfuse 上报封装。所有方法失败时静默，不抛异常。"""

    def __init__(self):
        self._enabled = False
        self._langfuse = None
        self._traces: Dict[str, Any] = {}
        pk = os.environ.get("LANGFUSE_PUBLIC_KEY")
        sk = os.environ.get("LANGFUSE_SECRET_KEY")
        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        if pk and sk:
            try:
                from langfuse import Langfuse
                self._langfuse = Langfuse(public_key=pk, secret_key=sk, host=host)
                self._enabled = True
            except Exception:
                pass

    @property
    def enabled(self) -> bool:
        return self._enabled

    def trace_task(self, task_id: str, industry: str, competitors: List[str]) -> None:
        if not self._enabled:
            return
        try:
            trace = self._langfuse.trace(
                id=task_id,
                name=f"竞品分析: {industry}",
                metadata={"industry": industry, "competitors": competitors},
            )
            self._traces[task_id] = trace
        except Exception:
            pass

    def span_agent(
        self,
        task_id: str,
        agent: str,
        round_no: int,
        input_payload: str,
        output_payload: str,
        status: str,
        latency_ms: int,
    ) -> None:
        if not self._enabled:
            return
        try:
            trace = self._traces.get(task_id)
            if not trace:
                return
            trace.span(
                name=f"{agent}_r{round_no}",
                metadata={"agent": agent, "round": round_no, "status": status},
                input=input_payload,
                output=output_payload,
                level="ERROR" if status == "err" else "DEFAULT",
            )
        except Exception:
            pass

    def generation(
        self,
        task_id: str,
        agent: str,
        round_no: int,
        system: str,
        user: str,
        output: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: int,
        model: str = "doubao",
        status: str = "ok",
    ) -> None:
        if not self._enabled:
            return
        try:
            trace = self._traces.get(task_id)
            if not trace:
                return
            trace.generation(
                name=f"{agent}_llm_r{round_no}",
                model=model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                output=output,
                usage={"input": tokens_in, "output": tokens_out},
                metadata={"latency_ms": latency_ms, "status": status},
                level="ERROR" if status == "err" else "DEFAULT",
            )
        except Exception:
            pass

    def flush(self) -> None:
        if self._enabled and self._langfuse:
            try:
                self._langfuse.flush()
            except Exception:
                pass