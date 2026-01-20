"""
文本分块器基类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from kg_search.ingestion.loaders.base import Document


@dataclass
class Chunk:
    """文本块数据结构"""

    id: str
    content: str
    document_id: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    # 继承自文档的字段
    artifact_id: str | None = None
    artifact_name: str | None = None
    dynasty: str | None = None

    # 块的位置信息
    start_char: int = 0
    end_char: int = 0

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
            "artifact_id": self.artifact_id,
            "artifact_name": self.artifact_name,
            "dynasty": self.dynasty,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


class TextChunker(ABC):
    """文本分块器抽象基类"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        初始化分块器

        Args:
            chunk_size: 块大小（字符数）
            chunk_overlap: 块重叠（字符数）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """
        将文档分块

        Args:
            document: 输入文档

        Returns:
            块列表
        """
        pass

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """
        批量处理文档

        Args:
            documents: 文档列表

        Returns:
            所有块的列表
        """
        chunks = []
        for doc in documents:
            chunks.extend(self.chunk(doc))
        return chunks
