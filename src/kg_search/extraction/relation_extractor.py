"""
关系抽取器

使用LLM从文本中提取实体间关系
"""

import json
from enum import Enum
from typing import Any

from kg_search.utils import generate_id, get_logger

from .models import Entity, Relation
from .prompts.relation_prompt import RELATION_EXTRACTION_PROMPT, RELATION_TYPES
from .types import RelationType

logger = get_logger(__name__)


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
        entities_str = "\n".join([f"- {e.name} ({e.type_value})" for e in entities])

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

            # 构建实体名称到实体的映射
            entity_map = {e.name: e for e in entities}

            relations = []
            for data in relations_data:
                source_name = data.get("source_entity", "")
                target_name = data.get("target_entity", "")

                source_entity = entity_map.get(source_name)
                target_entity = entity_map.get(target_name)

                relation = Relation(
                    id=generate_id("relation"),
                    source_entity_id=source_entity.id if source_entity else "",
                    source_entity_name=source_name,
                    target_entity_id=target_entity.id if target_entity else "",
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
            type_key = entity.type_value
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
                        Relation.from_entities(
                            source=artifact,
                            target=dynasty,
                            relation_type=RelationType.BELONGS_TO_DYNASTY,
                            source_doc_id=source_doc_id,
                        )
                    )

            # 文物 -> 材质
            for material in entities_by_type.get("材质", []):
                if artifact.attributes.get("material") == material.name:
                    relations.append(
                        Relation.from_entities(
                            source=artifact,
                            target=material,
                            relation_type=RelationType.MADE_OF,
                            source_doc_id=source_doc_id,
                        )
                    )

            # 文物 -> 地点
            for location in entities_by_type.get("地点", []):
                if artifact.attributes.get("location") == location.name:
                    relations.append(
                        Relation.from_entities(
                            source=artifact,
                            target=location,
                            relation_type=RelationType.EXCAVATED_FROM,
                            source_doc_id=source_doc_id,
                        )
                    )

            # 文物 -> 博物馆
            for museum in entities_by_type.get("收藏机构", []):
                relations.append(
                    Relation.from_entities(
                        source=artifact,
                        target=museum,
                        relation_type=RelationType.COLLECTED_BY,
                        source_doc_id=source_doc_id,
                    )
                )

            # 文物 -> 文化
            for culture in entities_by_type.get("文化", []):
                relations.append(
                    Relation.from_entities(
                        source=artifact,
                        target=culture,
                        relation_type=RelationType.BELONGS_TO_CULTURE,
                        source_doc_id=source_doc_id,
                    )
                )

        return relations
