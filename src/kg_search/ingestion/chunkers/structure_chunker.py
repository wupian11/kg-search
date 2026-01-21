"""
结构化分块器

专门处理结构化数据（JSON、字典等），保持数据完整性
"""

from typing import Any, Literal

from kg_search.ingestion.loaders.base import Document
from kg_search.utils import count_tokens

from .base import Chunk, TextChunker


class StructureChunker(TextChunker):
    """
    结构化分块器

    专门处理结构化数据，适用于：
    - JSON数据（如文化遗产记录）
    - 字典格式数据
    - 有明确字段的结构化文档

    策略：
    - record: 按记录分块，每条记录一个块（适合短记录）
    - field: 按字段组分块（适合长记录）
    - auto: 自动选择策略
    """

    # 默认字段组配置
    DEFAULT_FIELD_GROUPS = [
        # 核心标识字段组
        ["name", "caseId", "classId", "level"],
        # 分类描述字段组
        ["typeName", "category", "type"],
        # 地理位置字段组
        ["cityName", "distName", "address", "longitude", "latitude"],
        # 详细描述字段组
        ["registerReason", "reserveStatusDesc", "description"],
        # 时间信息字段组
        ["dynasty", "createTime", "updateTime"],
        # 媒体链接字段组
        ["imageUrl", "url", "links"],
    ]

    # 上下文字段（每个块都应包含）
    DEFAULT_CONTEXT_FIELDS = ["name", "caseId"]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 0,  # 结构化数据通常不需要重叠
        max_tokens: int = 500,
        strategy: Literal["record", "field", "auto"] = "auto",
        field_groups: list[list[str]] | None = None,
        context_fields: list[str] | None = None,
        include_field_names: bool = True,
    ):
        """
        初始化结构化分块器

        Args:
            chunk_size: 目标块大小（字符数）
            chunk_overlap: 块重叠（结构化数据通常为0）
            max_tokens: 最大token数
            strategy: 分块策略
                - record: 每条记录一个块
                - field: 按字段组分块
                - auto: 自动选择
            field_groups: 字段组定义，每组字段会尝试放在同一块中
            context_fields: 上下文字段，会添加到每个块中
            include_field_names: 是否在内容中包含字段名
        """
        super().__init__(chunk_size, chunk_overlap)
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.field_groups = field_groups or self.DEFAULT_FIELD_GROUPS
        self.context_fields = context_fields or self.DEFAULT_CONTEXT_FIELDS
        self.include_field_names = include_field_names

    def chunk(self, document: Document) -> list[Chunk]:
        """将文档分块"""
        content = document.content

        if not content:
            return []

        # 尝试获取结构化数据
        structured_data = self._extract_structured_data(document)

        if structured_data:
            # 使用结构化分块
            return self._chunk_structured(document, structured_data)
        else:
            # 回退到简单分块
            return self._chunk_plain_text(document)

    def _extract_structured_data(self, document: Document) -> dict | None:
        """从文档提取结构化数据"""
        import json

        # 优先从metadata获取原始数据
        if document.metadata:
            # 检查是否有原始JSON数据
            if "raw_data" in document.metadata:
                return document.metadata["raw_data"]
            if "original_data" in document.metadata:
                return document.metadata["original_data"]

        # 尝试解析content为JSON
        try:
            data = json.loads(document.content)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, TypeError):
            pass

        return None

    def _chunk_structured(self, document: Document, data: dict) -> list[Chunk]:
        """对结构化数据进行分块"""
        strategy = self._determine_strategy(data)

        if strategy == "record":
            return self._chunk_as_record(document, data)
        else:
            return self._chunk_by_fields(document, data)

    def _determine_strategy(self, data: dict) -> str:
        """自动确定分块策略"""
        if self.strategy != "auto":
            return self.strategy

        # 计算整体大小
        full_text = self._format_data(data)
        token_count = count_tokens(full_text)

        # 如果整体较小，使用记录策略
        if token_count <= self.max_tokens:
            return "record"
        else:
            return "field"

    def _chunk_as_record(self, document: Document, data: dict) -> list[Chunk]:
        """整条记录作为一个块"""
        content = self._format_data(data)

        chunk = Chunk(
            id=f"{document.id}_chunk_0",
            content=content,
            document_id=document.id,
            chunk_index=0,
            metadata={
                **document.metadata,
                "chunk_strategy": "record",
                "field_count": len(data),
            },
            artifact_id=document.artifact_id,
            artifact_name=document.artifact_name,
            dynasty=document.dynasty,
            start_char=0,
            end_char=len(content),
        )

        return [chunk]

    def _chunk_by_fields(self, document: Document, data: dict) -> list[Chunk]:
        """按字段组分块"""
        chunks = []

        # 准备上下文信息
        context = self._build_context(data)

        # 按字段组处理
        used_fields = set(self.context_fields)
        chunk_index = 0

        for field_group in self.field_groups:
            group_data = {}
            for field in field_group:
                if field in data and field not in used_fields:
                    group_data[field] = data[field]
                    used_fields.add(field)

            if group_data:
                # 格式化字段组内容
                group_content = self._format_data(group_data)

                # 检查是否需要添加上下文
                if context:
                    full_content = f"{context}\n---\n{group_content}"
                else:
                    full_content = group_content

                # 检查大小，如果过大则拆分
                if count_tokens(full_content) > self.max_tokens:
                    # 逐字段处理
                    for field, value in group_data.items():
                        field_content = self._format_field(field, value)
                        if context:
                            field_content = f"{context}\n---\n{field_content}"

                        chunk = self._create_chunk(document, field_content, chunk_index, [field])
                        chunks.append(chunk)
                        chunk_index += 1
                else:
                    chunk = self._create_chunk(
                        document, full_content, chunk_index, list(group_data.keys())
                    )
                    chunks.append(chunk)
                    chunk_index += 1

        # 处理剩余字段
        remaining_data = {k: v for k, v in data.items() if k not in used_fields}
        if remaining_data:
            remaining_content = self._format_data(remaining_data)
            if context:
                remaining_content = f"{context}\n---\n{remaining_content}"

            chunk = self._create_chunk(
                document, remaining_content, chunk_index, list(remaining_data.keys())
            )
            chunks.append(chunk)

        return chunks

    def _build_context(self, data: dict) -> str:
        """构建上下文信息"""
        context_data = {}
        for field in self.context_fields:
            if field in data:
                context_data[field] = data[field]

        if context_data:
            return self._format_data(context_data)
        return ""

    def _format_data(self, data: dict) -> str:
        """格式化结构化数据为文本"""
        lines = []
        for key, value in data.items():
            lines.append(self._format_field(key, value))
        return "\n".join(lines)

    def _format_field(self, key: str, value: Any) -> str:
        """格式化单个字段"""
        if value is None:
            return ""

        if isinstance(value, (list, dict)):
            import json

            value_str = json.dumps(value, ensure_ascii=False)
        else:
            value_str = str(value)

        if self.include_field_names:
            return f"{key}: {value_str}"
        else:
            return value_str

    def _create_chunk(
        self,
        document: Document,
        content: str,
        chunk_index: int,
        fields: list[str],
    ) -> Chunk:
        """创建Chunk对象"""
        return Chunk(
            id=f"{document.id}_chunk_{chunk_index}",
            content=content,
            document_id=document.id,
            chunk_index=chunk_index,
            metadata={
                **document.metadata,
                "chunk_strategy": "field",
                "fields": fields,
            },
            artifact_id=document.artifact_id,
            artifact_name=document.artifact_name,
            dynasty=document.dynasty,
            start_char=0,
            end_char=len(content),
        )

    def _chunk_plain_text(self, document: Document) -> list[Chunk]:
        """回退：处理纯文本"""
        content = document.content

        # 如果够小，直接返回一个块
        if count_tokens(content) <= self.max_tokens:
            return [
                Chunk(
                    id=f"{document.id}_chunk_0",
                    content=content,
                    document_id=document.id,
                    chunk_index=0,
                    metadata={**document.metadata, "chunk_strategy": "plain"},
                    artifact_id=document.artifact_id,
                    artifact_name=document.artifact_name,
                    dynasty=document.dynasty,
                    start_char=0,
                    end_char=len(content),
                )
            ]

        # 简单按大小分割
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            chunk_content = content[start:end]

            chunks.append(
                Chunk(
                    id=f"{document.id}_chunk_{chunk_index}",
                    content=chunk_content,
                    document_id=document.id,
                    chunk_index=chunk_index,
                    metadata={**document.metadata, "chunk_strategy": "plain"},
                    artifact_id=document.artifact_id,
                    artifact_name=document.artifact_name,
                    dynasty=document.dynasty,
                    start_char=start,
                    end_char=end,
                )
            )

            start = end
            chunk_index += 1

        return chunks
