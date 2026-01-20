"""
文档加载器基类

定义文档数据结构和加载器接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Document:
    """文档数据结构"""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    doc_type: str = ""

    # 文博领域特有字段（从嵌套JSON中提取）
    artifact_id: str | None = None
    artifact_name: str | None = None
    dynasty: str | None = None
    period: str | None = None
    material: str | None = None
    technique: str | None = None
    location: str | None = None
    museum: str | None = None
    culture: str | None = None
    dimensions: str | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "source": self.source,
            "doc_type": self.doc_type,
            "artifact_id": self.artifact_id,
            "artifact_name": self.artifact_name,
            "dynasty": self.dynasty,
            "period": self.period,
            "material": self.material,
            "technique": self.technique,
            "location": self.location,
            "museum": self.museum,
            "culture": self.culture,
            "dimensions": self.dimensions,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        """从字典创建文档"""
        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            source=data.get("source", ""),
            doc_type=data.get("doc_type", ""),
            artifact_id=data.get("artifact_id"),
            artifact_name=data.get("artifact_name"),
            dynasty=data.get("dynasty"),
            period=data.get("period"),
            material=data.get("material"),
            technique=data.get("technique"),
            location=data.get("location"),
            museum=data.get("museum"),
            culture=data.get("culture"),
            dimensions=data.get("dimensions"),
            description=data.get("description"),
        )


class DocumentLoader(ABC):
    """文档加载器抽象基类"""

    supported_extensions: list[str] = []

    @abstractmethod
    def load(self, file_path: str | Path) -> list[Document]:
        """
        加载文档

        Args:
            file_path: 文件路径

        Returns:
            文档列表
        """
        pass

    @abstractmethod
    def load_from_string(self, content: str, source: str = "") -> list[Document]:
        """
        从字符串加载文档

        Args:
            content: 文档内容字符串
            source: 来源标识

        Returns:
            文档列表
        """
        pass

    def can_load(self, file_path: str | Path) -> bool:
        """
        检查是否支持加载该文件

        Args:
            file_path: 文件路径

        Returns:
            是否支持
        """
        path = Path(file_path)
        return path.suffix.lower() in self.supported_extensions
