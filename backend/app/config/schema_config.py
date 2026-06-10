"""
动态 Schema 配置：运行时控制竞品分析的字段、topic、校验规则。
通过 PUT /api/schema 可热更新，无需重启服务。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_CONFIG_PATH = Path(__file__).parent / "schema_config.json"


class SchemaConfig:
    _instance: Optional["SchemaConfig"] = None

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self._data: Dict[str, Any] = {}
        self.reload()

    @classmethod
    def get(cls) -> "SchemaConfig":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reload(self) -> None:
        if self.config_path.exists():
            self._data = json.loads(self.config_path.read_text(encoding="utf-8"))
        else:
            self._data = {}

    def update(self, new_data: Dict[str, Any]) -> None:
        self._data.update(new_data)
        self.config_path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def get_active_fields(self) -> List[Dict[str, Any]]:
        return self._data.get("competitor_fields", [])

    def get_topics(self) -> List[str]:
        return self._data.get("evidence_topics", [])

    def topic_for_field(self, field: str) -> Optional[str]:
        return self._data.get("field_to_topic", {}).get(field)

    def field_for_topic(self, topic: str) -> Optional[str]:
        return self._data.get("topic_to_field", {}).get(topic)

    def get_swot_quadrants(self) -> List[Dict[str, str]]:
        return self._data.get("swot_quadrants", [])

    def get_writer_sections(self) -> List[str]:
        return self._data.get("writer_sections", [])

    def get_qa_required_fields(self) -> List[str]:
        return self._data.get("qa_required_fields", [])