"""
JSON/JSONL文档加载器

支持嵌套结构的文物数据JSON格式
"""

import json
from pathlib import Path
from typing import Any

from kg_search.utils import extract_nested_value, generate_id

from .base import Document, DocumentLoader


class JSONLoader(DocumentLoader):
    """JSON文档加载器"""

    supported_extensions = [".json"]

    # 嵌套JSON字段映射配置
    # 格式: 目标字段 -> 嵌套路径
    DEFAULT_FIELD_MAPPING = {
        "artifact_id": "id",
        "artifact_name": "basic_info.name",
        "dynasty": "basic_info.dynasty",
        "period": "basic_info.period",
        "material": "physical_info.material",
        "technique": "physical_info.technique",
        "dimensions": "physical_info.dimensions",
        "location": "provenance.excavation_site",
        "museum": "provenance.museum",
        "culture": "basic_info.culture",
        "description": "description",
    }

    def __init__(self, field_mapping: dict[str, str] | None = None):
        """
        初始化JSON加载器

        Args:
            field_mapping: 自定义字段映射，键为Document字段名，值为JSON中的路径
        """
        self.field_mapping = field_mapping or self.DEFAULT_FIELD_MAPPING

    def load(self, file_path: str | Path) -> list[Document]:
        """加载JSON文件"""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self._parse_data(data, source=str(path))

    def load_from_string(self, content: str, source: str = "") -> list[Document]:
        """从JSON字符串加载"""
        data = json.loads(content)
        return self._parse_data(data, source=source)

    def _parse_data(self, data: Any, source: str = "") -> list[Document]:
        """
        解析JSON数据

        Args:
            data: JSON数据（可以是单个对象或数组）
            source: 来源标识

        Returns:
            文档列表
        """
        if isinstance(data, list):
            return [self._create_document(item, source) for item in data]
        elif isinstance(data, dict):
            # 检查是否有artifacts数组
            if "artifacts" in data:
                return [self._create_document(item, source) for item in data["artifacts"]]
            return [self._create_document(data, source)]
        else:
            raise ValueError(f"Unsupported JSON structure: {type(data)}")

    def _create_document(self, data: dict[str, Any], source: str) -> Document:
        """
        从嵌套JSON创建文档

        Args:
            data: 单个文物的JSON数据
            source: 来源

        Returns:
            Document对象
        """
        # 提取字段值
        extracted = {}
        for doc_field, json_path in self.field_mapping.items():
            value = extract_nested_value(data, json_path)
            if value is not None:
                extracted[doc_field] = value

        # 生成文档内容（用于向量化）
        content = self._build_content(extracted, data)

        # 生成ID
        doc_id = extracted.get("artifact_id") or generate_id("artifact")

        return Document(
            id=doc_id,
            content=content,
            metadata=data,  # 保存原始完整数据
            source=source,
            doc_type="artifact",
            artifact_id=extracted.get("artifact_id"),
            artifact_name=extracted.get("artifact_name"),
            dynasty=extracted.get("dynasty"),
            period=extracted.get("period"),
            material=extracted.get("material"),
            technique=extracted.get("technique"),
            location=extracted.get("location"),
            museum=extracted.get("museum"),
            culture=extracted.get("culture"),
            dimensions=extracted.get("dimensions"),
            description=extracted.get("description"),
        )

    def _build_content(self, extracted: dict[str, Any], raw_data: dict[str, Any]) -> str:
        """
        构建用于向量化的文档内容

        Args:
            extracted: 提取的字段
            raw_data: 原始数据

        Returns:
            格式化的文本内容
        """
        parts = []

        if name := extracted.get("artifact_name"):
            parts.append(f"文物名称：{name}")

        if dynasty := extracted.get("dynasty"):
            parts.append(f"朝代：{dynasty}")

        if period := extracted.get("period"):
            parts.append(f"年代：{period}")

        if culture := extracted.get("culture"):
            parts.append(f"文化：{culture}")

        if material := extracted.get("material"):
            parts.append(f"材质：{material}")

        if technique := extracted.get("technique"):
            parts.append(f"工艺：{technique}")

        if dimensions := extracted.get("dimensions"):
            parts.append(f"尺寸：{dimensions}")

        if location := extracted.get("location"):
            parts.append(f"出土地点：{location}")

        if museum := extracted.get("museum"):
            parts.append(f"收藏机构：{museum}")

        if description := extracted.get("description"):
            parts.append(f"描述：{description}")

        return "\n".join(parts)


class JSONLLoader(DocumentLoader):
    """JSONL文档加载器"""

    supported_extensions = [".jsonl"]

    def __init__(self, field_mapping: dict[str, str] | None = None):
        """初始化JSONL加载器"""
        self.json_loader = JSONLoader(field_mapping=field_mapping)

    def load(self, file_path: str | Path) -> list[Document]:
        """加载JSONL文件"""
        path = Path(file_path)
        documents = []

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        docs = self.json_loader._parse_data(data, source=f"{path}:L{line_num}")
                        documents.extend(docs)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON at line {line_num}: {e}")

        return documents

    def load_from_string(self, content: str, source: str = "") -> list[Document]:
        """从JSONL字符串加载"""
        documents = []
        for line_num, line in enumerate(content.split("\n"), 1):
            line = line.strip()
            if line:
                data = json.loads(line)
                docs = self.json_loader._parse_data(data, source=f"{source}:L{line_num}")
                documents.extend(docs)
        return documents
