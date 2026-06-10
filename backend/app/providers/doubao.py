"""
Doubao（火山方舟 ARK）LLM Provider — OpenAI 兼容协议。

文档: https://www.volcengine.com/docs/82379

用法:
provider = DoubaoLLMProvider(
    api_key="ark-xxx",
    endpoint_id="ep-202xxx",  # EP（推理接入点 ID），作为 model 字段传入
)
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from app.providers.llm import LLMProvider, LLMResponse

DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


class DoubaoLLMProvider(LLMProvider):
    """通过 openai SDK 调用方舟 EP。"""

    def __init__(
        self,
        api_key: str,
        endpoint_id: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        max_tokens: int = 4096,
        timeout: float = 60.0,
    ):
        # 延迟导入：仅当真实启用 Doubao 时才依赖 openai
        from openai import OpenAI

        if not api_key:
            raise ValueError("DoubaoLLMProvider 需要 api_key")
        if not endpoint_id:
            raise ValueError("DoubaoLLMProvider 需要 endpoint id (EP)")

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.endpoint_id = endpoint_id
        self.max_tokens = max_tokens

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> LLMResponse:
        start = time.time()
        # json_mode 提示：通过 prompt 强约束（Doubao-Seed-2.0-lite 不支持 response_format=json_object）
        if json_mode:
            user = user + "\n\n请直接输出 JSON，不要任何前后缀文字，不要 markdown 代码块。"

        kwargs: Dict[str, Any] = {
            "model": self.endpoint_id,  # ~ 方舟用 EP ID 作为 model 字段传入
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        resp = self.client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        if json_mode:
            text = _extract_json(text)

        usage = getattr(resp, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", 0) if usage else 0
        tokens_out = getattr(usage, "completion_tokens", 0) if usage else 0

        return LLMResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=int((time.time() - start) * 1000),
            raw={
                "id": getattr(resp, "id", None),
                "model": getattr(resp, "model", None),
            },
        )


def _extract_json(text: str) -> str:
    """从 LLM 输出中抽取 JSON 子串。

    处理三类情况:
    1. 直接是 JSON → 原样返回
    2. ```json ... ``` 代码块 → 去掉围栏
    3. 有前后缀文字 → 截取首个 `{` 到最后一个 `}` 之间
    """
    t = text.strip()
    # 代码块
    if t.startswith("```"):
        t = t.strip("`")
        if t.startswith("json"):
            t = t[4:]
        t = t.strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    # 首尾大括号兜底
    if not t.startswith("{"):
        l = t.find("{")
        r = t.rfind("}")
        if l != -1 and r != -1 and r > l:
            t = t[l : r + 1]
    return t