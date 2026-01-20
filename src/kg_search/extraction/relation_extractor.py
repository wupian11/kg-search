"""
关系抽取器

使用LLM从文本中提取实体间关系
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kg_search.utils import generate_id, get_logger

from .entity_extractor import Entity
from .prompts.relation_prompt import RELATION_EXTRACTION_PROMPT, RELATION_TYPES

logger = get_logger(__name__)


class RelationType(str, Enum):
    """文博领域关系类型"""

    BELONGS_TO_DYNASTY = "属于朝代"
    EXCAVATED_FROM = "出土于"
    COLLECTED_BY = "收藏于"
    MADE_OF = "材质为"
    CREATED_BY = "制作者"
    DISCOVERED_BY = "发现者"
    SAME_PERIOD = "同时期"
    SIMILAR_STYLE = "风格相似"
    TECHNIQUE_INHERIT = "工艺传承"
    BELONGS_TO_CULTURE = "属于文化"
    LOCATED_IN = "位于"
    RELATED_TO = "相关"


@dataclass
class Relation:
    """关系数据结构"""

    id: str
    source_entity_id: str
    source_entity_name: str
    target_entity_id: str
    target_entity_name: str
    relation_type: RelationType | str
    description: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    confidence: float = 1.0
    source_doc_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "source_entity_id": self.source_entity_id,
            "source_entity_name": self.source_entity_name,
            "target_entity_id": self.target_entity_id,
            "target_entity_name": self.target_entity_name,
            "relation_type": self.relation_type.value
            if isinstance(self.relation_type, RelationType)
            else self.relation_type,
            "description": self.description,
            "attributes": self.attributes,
            "weight": self.weight,
            "confidence": self.confidence,
            "source_doc_id": self.source_doc_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Relation":
        """从字典创建关系"""
        rel_type = data.get("relation_type", "")
        try:
            rel_type = RelationType(rel_type)
        except ValueError:
            pass

        return cls(
            id=data.get("id", generate_id("relation")),
            source_entity_id=data.get("source_entity_id", ""),
            source_entity_name=data.get("source_entity_name", ""),
            target_entity_id=data.get("target_entity_id", ""),
            target_entity_name=data.get("target_entity_name", ""),
            relation_type=rel_type,
            description=data.get("description", ""),
            attributes=data.get("attributes", {}),
            weight=data.get("weight", 1.0),
            confidence=data.get("confidence", 1.0),
            source_doc_id=data.get("source_doc_id", ""),
        )


class RelationExtractor:
    """关系抽取器"""

    def __init__(self, llm_client: Any):
        """
        初始化关系抽取器

        Args:
            llm_client: LLM客户端实例
        """
        self.llm_client = llm_client

    async def extract_from_text(
        self,
        text: str,
        entities: list[Entity],
        source_doc_id: str = "",
    ) -> list[Relation]:
        """
        从文本中提取实体间关系

        Args:
            text: 输入文本
            entities: 已提取的实体列表
            source_doc_id: 来源文档ID

        Returns:
            关系列表
        """
        if len(entities) < 2:
            return []

        # 构建实体列表字符串
        entities_str = "\n".join(
            [
                f"- {e.name} ({e.type.value if isinstance(e.type, Enum) else e.type})"
                for e in entities
            ]
        )

        prompt = RELATION_EXTRACTION_PROMPT.format(
            relation_types=RELATION_TYPES,
            entities=entities_str,
            text=text,
        )

        try:
            response = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            result = json.loads(response)
            relations_data = result.get("relations", [])

            # 构建实体名称到ID的映射
            entity_name_to_id = {e.name: e.id for e in entities}

            relations = []
            for data in relations_data:
                # 查找实体ID
                source_name = data.get("source_entity", "")
                target_name = data.get("target_entity", "")

                relation = Relation(
                    id=generate_id("relation"),
                    source_entity_id=entity_name_to_id.get(source_name, ""),
                    source_entity_name=source_name,
                    target_entity_id=entity_name_to_id.get(target_name, ""),
                    target_entity_name=target_name,
                    relation_type=data.get("relation_type", "相关"),
                    description=data.get("description", ""),
                    confidence=data.get("confidence", 1.0),
                    source_doc_id=source_doc_id,
                )
                relations.append(relation)

            logger.info("Extracted relations", count=len(relations))
            return relations

        except Exception as e:
            logger.error("Relation extraction failed", error=str(e))
            return []

    def extract_structured_relations(
        self,
        entities: list[Entity],
        source_doc_id: str = "",
    ) -> list[Relation]:
        """
        从结构化实体中提取隐含关系

        Args:
            entities: 实体列表
            source_doc_id: 来源文档ID

        Returns:
            关系列表
        """
        relations = []

        # 按类型分组实体
        entities_by_type: dict[str, list[Entity]] = {}
        for entity in entities:
            type_key = entity.type.value if isinstance(entity.type, Enum) else entity.type
            if type_key not in entities_by_type:
                entities_by_type[type_key] = []
            entities_by_type[type_key].append(entity)

        # 文物实体
        artifacts = entities_by_type.get("文物", [])

        for artifact in artifacts:
            # 文物 -> 朝代
            for dynasty in entities_by_type.get("朝代", []):
                if artifact.attributes.get("dynasty") == dynasty.name:
                    relations.append(
                        Relation(
                            id=generate_id("rel"),
                            source_entity_id=artifact.id,
                            source_entity_name=artifact.name,
                            target_entity_id=dynasty.id,
                            target_entity_name=dynasty.name,
                            relation_type=RelationType.BELONGS_TO_DYNASTY,
                            source_doc_id=source_doc_id,
                        )
                    )

            # 文物 -> 材质
            for material in entities_by_type.get("材质", []):
                if artifact.attributes.get("material") == material.name:
                    relations.append(
                        Relation(
                            id=generate_id("rel"),
                            source_entity_id=artifact.id,
                            source_entity_name=artifact.name,
                            target_entity_id=material.id,
                            target_entity_name=material.name,
                            relation_type=RelationType.MADE_OF,
                            source_doc_id=source_doc_id,
                        )
                    )

            # 文物 -> 地点
            for location in entities_by_type.get("地点", []):
                if artifact.attributes.get("location") == location.name:
                    relations.append(
                        Relation(
                            id=generate_id("rel"),
                            source_entity_id=artifact.id,
                            source_entity_name=artifact.name,
                            target_entity_id=location.id,
                            target_entity_name=location.name,
                            relation_type=RelationType.EXCAVATED_FROM,
                            source_doc_id=source_doc_id,
                        )
                    )

            # 文物 -> 博物馆
            for museum in entities_by_type.get("收藏机构", []):
                relations.append(
                    Relation(
                        id=generate_id("rel"),
                        source_entity_id=artifact.id,
                        source_entity_name=artifact.name,
                        target_entity_id=museum.id,
                        target_entity_name=museum.name,
                        relation_type=RelationType.COLLECTED_BY,
                        source_doc_id=source_doc_id,
                    )
                )

            # 文物 -> 文化
            for culture in entities_by_type.get("文化", []):
                relations.append(
                    Relation(
                        id=generate_id("rel"),
                        source_entity_id=artifact.id,
                        source_entity_name=artifact.name,
                        target_entity_id=culture.id,
                        target_entity_name=culture.name,
                        relation_type=RelationType.BELONGS_TO_CULTURE,
                        source_doc_id=source_doc_id,
                    )
                )

        return relations
