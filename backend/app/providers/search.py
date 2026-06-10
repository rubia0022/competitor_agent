"""
Search Provider 抽象：信息采集 Agent 的数据入口。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List

from app.schemas import RawEvidence


class SearchProvider(ABC):
    @abstractmethod
    def search(self, competitor: str, industry: str) -> List[RawEvidence]:
        ...


class MockSearchProvider(SearchProvider):
    """从 `data/mock/{competitor}.json` 读取离线证据。
    JSON 结构示例见 `data/mock/feishu.json`。
    """

    def __init__(self, mock_dir: Path):
        self.mock_dir = Path(mock_dir)

    def search(self, competitor: str, industry: str) -> List[RawEvidence]:
        # 文件名归一化：去空格、转小写、做映射
        candidates = [
            self.mock_dir / f"{competitor}.json",
            self.mock_dir / f"{competitor.lower()}.json",
            self.mock_dir / f"{self._slug(competitor)}.json",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return [
            RawEvidence(
                competitor=competitor,
                topic=item["topic"],
                content=item["content"],
                source_url=item["source_url"],
                fetched_at=datetime.fromisoformat(
                    item.get("fetched_at", datetime.utcnow().isoformat())
                ),
                confidence=item.get("confidence", 0.75),
            )
            for item in data
        ]

    @staticmethod
    def _slug(name: str) -> str:
        mapping = {
            "飞书": "feishu",
            "钉钉": "dingtalk",
            "企业微信": "wecom",
        }
        return mapping.get(name, name.lower().replace(" ", "_"))