"""
核心数据 Schema：竞品知识结构 + Citation 强制溯源。
"""

from __future__ import annotations

from datetime import datetime
from typing import List

from pydantic import BaseModel, Field


def _list_min_constraint(min_n: int) -> dict:
    try:
        from pydantic.version import VERSION as _V
    except Exception:
        from pydantic import __version__ as _V  # type: ignore
    major = int(str(_V).split(".", 1)[0])
    if major >= 2:
        return {"min_length": min_n}
    return {"min_items": min_n}


_CITATIONS_MIN_1 = _list_min_constraint(1)


class Citation(BaseModel):
    """溯源引用：每条原子结论都必须带至少一个 Citation。"""
    source_url: str = Field(..., description="原始信息源 URL")
    snippet: str = Field(..., min_length=1, description="原文片段，用于核验")
    fetched_at: datetime = Field(..., description="抓取时间")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度 0-1")


class FeatureNode(BaseModel):
    """功能树节点，支持嵌套。"""
    name: str
    description: str = ""
    children: List["FeatureNode"] = Field(default_factory=list)
    citations: List[Citation] = Field(..., **_CITATIONS_MIN_1)


if hasattr(FeatureNode, "model_rebuild"):
    FeatureNode.model_rebuild()
else:
    FeatureNode.update_forward_refs()


class PricingPlan(BaseModel):
    tier_name: str
    price: str = Field(..., description="字符串以兼容‘联系销售’等非数值情形")
    features_included: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(..., **_CITATIONS_MIN_1)


class Persona(BaseModel):
    segment: str = Field(..., description="用户画像段，如‘中大型企业 HR’")
    pain_points: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(..., **_CITATIONS_MIN_1)


class ClaimWithCitation(BaseModel):
    """一条带引用的结论，用于 SWOT。"""
    claim: str
    citations: List[Citation] = Field(..., **_CITATIONS_MIN_1)


class SWOT(BaseModel):
    strengths: List[ClaimWithCitation] = Field(default_factory=list)
    weaknesses: List[ClaimWithCitation] = Field(default_factory=list)
    opportunities: List[ClaimWithCitation] = Field(default_factory=list)
    threats: List[ClaimWithCitation] = Field(default_factory=list)


class Competitor(BaseModel):
    """单个竞品的完整结构化描述。"""
    name: str
    features: List[FeatureNode] = Field(default_factory=list)
    pricing: List[PricingPlan] = Field(default_factory=list)
    personas: List[Persona] = Field(default_factory=list)
    swot: SWOT = Field(default_factory=SWOT)


class RawEvidence(BaseModel):
    """Collector 的原子输出：一段原始证据。"""
    competitor: str
    topic: str = Field(..., description="证据分类，由 SchemaConfig 控制合法值")
    content: str
    source_url: str
    fetched_at: datetime
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class CompetitorReport(BaseModel):
    """Writer 的最终输出：报告 = 多竞品结构化数据 + Markdown。"""
    title: str
    industry: str
    competitors: List[Competitor]
    markdown: str
    generated_at: datetime
