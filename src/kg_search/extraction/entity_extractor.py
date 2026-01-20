"""
实体抽取器

使用LLM从文本中提取文博领域实体
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kg_search.ingestion.chunkers.base import Chunk
from kg_search.ingestion.loaders.base import Document
from kg_search.utils import generate_id, get_logger

from .prompts.entity_prompt import ENTITY_EXTRACTION_PROMPT, ENTITY_TYPES

logger = get_logger(__name__)


class EntityType(str, Enum):
    """文博领域实体类型"""

    ARTIFACT = "文物"
    DYNASTY = "朝代"
    PERIOD = "年代"
    MATERIAL = "材质"
    TECHNIQUE = "工艺"
    LOCATION = "地点"
    MUSEUM = "收藏机构"
    PERSON = "人物"
    CULTURE = "文化"
    DIMENSION = "尺寸"
    STYLE = "风格"
    EVENT = "事件"


@dataclass
class Entity:
    """实体数据结构"""

    id: str
    name: str
    type: EntityType | str
    description: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    source_doc_id: str = ""
    source_chunk_id: str = ""
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value if isinstance(self.type, EntityType) else self.type,
            "description": self.description,
            "attributes": self.attributes,
            "source_doc_id": self.source_doc_id,
            "source_chunk_id": self.source_chunk_id,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entity":
        """从字典创建实体"""
        entity_type = data.get("type", "")
        try:
            entity_type = EntityType(entity_type)
        except ValueError:
            pass  # 保持字符串类型

        return cls(
            id=data.get("id", generate_id("entity")),
            name=data.get("name", ""),
            type=entity_type,
            description=data.get("description", ""),
            attributes=data.get("attributes", {}),
            source_doc_id=data.get("source_doc_id", ""),
            source_chunk_id=data.get("source_chunk_id", ""),
            confidence=data.get("confidence", 1.0),
        )


class EntityExtractor:
    """实体抽取器"""

    def __init__(self, llm_client: Any):
        """
        初始化实体抽取器

        Args:
            llm_client: LLM客户端实例
        """
        self.llm_client = llm_client

    async def extract_from_text(
        self,
        text: str,
        source_doc_id: str = "",
        source_chunk_id: str = "",
    ) -> list[Entity]:
        """
        从文本中提取实体

        Args:
            text: 输入文本
            source_doc_id: 来源文档ID
            source_chunk_id: 来源块ID

        Returns:
            实体列表
        """
        prompt = ENTITY_EXTRACTION_PROMPT.format(
            entity_types=ENTITY_TYPES,
            text=text,
        )

        try:
            response = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            # 解析响应
            result = json.loads(response)
            entities_data = result.get("entities", [])

            entities = []
            for data in entities_data:
                entity = Entity.from_dict(data)
                entity.source_doc_id = source_doc_id
                entity.source_chunk_id = source_chunk_id
                if not entity.id:
                    entity.id = generate_id(f"entity_{entity.type}")
                entities.append(entity)

            logger.info("Extracted entities", count=len(entities))
            return entities

        except Exception as e:
            logger.error("Entity extraction failed", error=str(e))
            return []

    async def extract_from_document(self, document: Document) -> list[Entity]:
        """从文档中提取实体"""
        # 从文档结构化字段直接创建实体
        entities = self._extract_structured_entities(document)

        # 从描述文本中用LLM提取额外实体
        if document.description:
            text_entities = await self.extract_from_text(
                document.description,
                source_doc_id=document.id,
            )
            entities.extend(text_entities)

        return self._deduplicate_entities(entities)

    async def extract_from_chunk(self, chunk: Chunk) -> list[Entity]:
        """从文本块中提取实体"""
        return await self.extract_from_text(
            chunk.content,
            source_doc_id=chunk.document_id,
            source_chunk_id=chunk.id,
        )

    def _extract_structured_entities(self, document: Document) -> list[Entity]:
        """从文档结构化字段提取实体"""
        entities = []

        # 文物实体
        if document.artifact_name:
            entities.append(
                Entity(
                    id=document.artifact_id or generate_id("artifact"),
                    name=document.artifact_name,
                    type=EntityType.ARTIFACT,
                    description=document.description or "",
                    attributes={
                        "dynasty": document.dynasty,
                        "material": document.material,
                        "location": document.location,
                    },
                    source_doc_id=document.id,
                )
            )

        # 朝代实体
        if document.dynasty:
            entities.append(
                Entity(
                    id=generate_id("dynasty"),
                    name=document.dynasty,
                    type=EntityType.DYNASTY,
                    source_doc_id=document.id,
                )
            )

        # 材质实体
        if document.material:
            entities.append(
                Entity(
                    id=generate_id("material"),
                    name=document.material,
                    type=EntityType.MATERIAL,
                    source_doc_id=document.id,
                )
            )

        # 地点实体
        if document.location:
            entities.append(
                Entity(
                    id=generate_id("location"),
                    name=document.location,
                    type=EntityType.LOCATION,
                    source_doc_id=document.id,
                )
            )

        # 博物馆实体
        if document.museum:
            entities.append(
                Entity(
                    id=generate_id("museum"),
                    name=document.museum,
                    type=EntityType.MUSEUM,
                    source_doc_id=document.id,
                )
            )

        # 文化实体
        if document.culture:
            entities.append(
                Entity(
                    id=generate_id("culture"),
                    name=document.culture,
                    type=EntityType.CULTURE,
                    source_doc_id=document.id,
                )
            )

        # 工艺实体
        if document.technique:
            entities.append(
                Entity(
                    id=generate_id("technique"),
                    name=document.technique,
                    type=EntityType.TECHNIQUE,
                    source_doc_id=document.id,
                )
            )

        return entities

    def _deduplicate_entities(self, entities: list[Entity]) -> list[Entity]:
        """实体去重"""
        seen = {}
        unique_entities = []

        for entity in entities:
            key = (entity.name.lower(), entity.type)
            if key not in seen:
                seen[key] = entity
                unique_entities.append(entity)
            else:
                # 合并属性
                existing = seen[key]
                existing.attributes.update(entity.attributes)

        return unique_entities
