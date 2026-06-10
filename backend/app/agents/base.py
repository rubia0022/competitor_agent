"""
Agent 基类：统一 trace 写入、异常处理、超时重试。
"""
from __future__ import annotations

import json
import random
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from app.providers import LLMProvider
from app.providers.llm import LLMResponse
from app.storage.trace import TraceStore

# — 重试策略 ————————————————————————————————————
RETRY_MAX_ATTEMPTS = 3           # 最多 3 次
RETRY_BASE_DELAY = 1.0           # 第一次失败后 1s
RETRY_MAX_DELAY = 10.0          # 单次重试间隔上限
RETRY_JITTER = 0.3              # ±30% 抖动，避免雪崩

# 视为可重试的错误关键字（覆盖 openai / httpx / 网络层）
RETRYABLE_KEYWORDS = {
    "Connection",
    "Timeout",
    "RateLimit",
    "ServiceUnavailable",
    "InternalServerError",
    "ReadTimeout",
    "APIConnectionError",
    "503",
    "504",
    "429",
}


def _is_retryable(exc: BaseException) -> bool:
    name = type(exc).__name__
    msg = str(exc)
    return any(k in name or k in msg for k in RETRYABLE_KEYWORDS)


def _backoff_delay(attempt: int) -> float:
    """指数退避 + 抖动。attempt 从 1 开始。"""
    base = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
    jitter = base * RETRY_JITTER * (random.random() * 2 - 1)
    return max(0.1, base + jitter)


class BaseAgent(ABC):
    role: str = "base"

    def __init__(self, llm: LLMProvider, trace_store: TraceStore):
        self.llm = llm
        self.trace = trace_store

    @abstractmethod
    def run(self, task_id: str, round_no: int, **kwargs) -> Dict[str, Any]:
        """子类实现具体业务。返回 dict [payload]。"""

    def call_llm(
        self,
        task_id: str,
        round_no: int,
        system: str,
        user: str,
        *,
        json_mode: bool = True,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """统一封装 LLM 调用 + 超时重试 + trace 落库 + JSON 解析。
        - 网络/限流/服务端错误：指数退避重试 RETRY_MAX_ATTEMPTS 次
        - JSON 解析失败：不重试，返回 {"_parse_error", "_raw"}
        - 所有尝试都失败：抛 RuntimeError 由上层降级
        """
        last_err: Optional[BaseException] = None
        resp: Optional[LLMResponse] = None
        total_start = time.time()
        attempts_log = []

        for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
            try:
                resp = self.llm.complete(
                    system=system, user=user,
                    json_mode=json_mode, temperature=temperature,
                )
                attempts_log.append(f"attempt{attempt}:ok")
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                retryable = _is_retryable(e)
                attempts_log.append(
                    f"attempt{attempt}:{type(e).__name__}({'retry' if retryable and attempt < RETRY_MAX_ATTEMPTS else 'give_up'})"
                )
                if not retryable or attempt >= RETRY_MAX_ATTEMPTS:
                    break
                delay = _backoff_delay(attempt)
                time.sleep(delay)

        total_latency = int((time.time() - total_start) * 1000)

        # 统一 trace 写入（一次调用一条记录，attempts 数量写在 output 前缀里）
        if last_err is not None:
            err_msg = f"{type(last_err).__name__}: {last_err}\n{traceback.format_exc()}"
            self.trace.write(
                task_id=task_id, agent=self.role, round_no=round_no,
                prompt=system, input_payload=user,
                output_payload=f"[{'|'.join(attempts_log)}]",
                tokens_in=0, tokens_out=0, latency_ms=total_latency,
                status="err", error_msg=err_msg,
            )
            raise RuntimeError(err_msg)

        assert resp is not None
        self.trace.write(
            task_id=task_id, agent=self.role, round_no=round_no,
            prompt=system, input_payload=user,
            output_payload=f"[{'|'.join(attempts_log)}] " + resp.text,
            tokens_in=resp.tokens_in, tokens_out=resp.tokens_out,
            latency_ms=total_latency, status="ok",
        )

        if self.trace.langfuse:
            self.trace.langfuse.generation(
                task_id=task_id, agent=self.role, round_no=round_no,
                system=system, user=user, output=resp.text,
                tokens_in=resp.tokens_in, tokens_out=resp.tokens_out,
                latency_ms=total_latency,
            )

        if json_mode:
            try:
                return json.loads(resp.text)
            except json.JSONDecodeError as e:
                return {"_parse_error": str(e), "_raw": resp.text}

        return {"text": resp.text}