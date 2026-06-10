"""
LLM Provider 抽象：上层 Agent 只依赖此接口，便于 Mock 与真实切换。

MVP 阶段：Agent 全部走确定性逻辑，不调 LLM（保证演示稳定）。
真实接入时：把对应 Agent 内 _build_competitor / _render 等替换为 self.call_llm(...)
即可。BaseAgent 已封装好 trace + JSON 解析。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMResponse:
    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    raw: Optional[Dict[str, Any]] = None


class LLMProvider(ABC):
    """所有 LLM 实现的统一入口。"""

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> LLMResponse:
        ...


class MockLLMProvider(LLMProvider):
    """演示用占位实现：返回空 JSON。
    在 MVP 中不会被实际调用（Agent 走确定性逻辑），仅为接口完整性预留。
    """

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> LLMResponse:
        start = time.time()
        text = "{}" if json_mode else ""
        return LLMResponse(
            text=text,
            tokens_in=len(system) + len(user),
            tokens_out=len(text),
            latency_ms=int((time.time() - start) * 1000),
            raw={"mock": True},
        )


# 真实 Anthropic LLM 实现示例（注释保留，便于后续切换）:
# class AnthropicLLMProvider(LLMProvider):
#     def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
#         from anthropic import Anthropic
#         self.client = Anthropic(api_key=api_key)
#         self.model = model
#
#     def complete(self, system, user, *, json_mode=False, temperature=0.2):
#         start = time.time()
#         resp = self.client.messages.create(
#             model=self.model,
#             system=system,
#             messages=[{"role": "user", "content": user}],
#             max_tokens=4096,
#             temperature=temperature,
#         )
#         return LLMResponse(
#             text=resp.content[0].text,
#             tokens_in=resp.usage.input_tokens,
#             tokens_out=resp.usage.output_tokens,
#             latency_ms=int((time.time() - start) * 1000),
#             raw=resp.model_dump(),
#         )