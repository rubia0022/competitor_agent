"""
Agent 间通信协议：所有跨 Agent 传递必须用 AgentMessage。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


AgentRole = Literal["collector", "analyst", "writer", "qa", "human", "orchestrator"]
Intent = Literal["produce", "review", "reject", "approve", "revise"]
Severity = Literal["low", "mid", "high"]


class Issue(BaseModel):
    """QA 找到的一个问题。"""
    field_path: str = Field(..., description="问题所在字段路径，如 competitors[0].pricing[1]")
    reason: str
    severity: Severity = "mid"


class QAReport(BaseModel):
    """质检报告。"""
    target_agent: AgentRole = Field(..., description="问题归属哪个上游 Agent")
    is_pass: bool
    issues: List[Issue] = Field(default_factory=list)
    summary: str = ""


class AgentMessage(BaseModel):
    """跨 Agent 通信的唯一载体。"""
    msg_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    from_agent: AgentRole
    to_agent: AgentRole
    intent: Intent
    payload: Dict[str, Any] = Field(default_factory=dict)
    parent_msg_id: Optional[str] = None
    round_no: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)