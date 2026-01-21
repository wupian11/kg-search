"""
GraphRAG抽取器适配

封装GraphRAG的实体和关系抽取功能
"""

import json
from typing import Any

from kg_search.extraction.entity_extractor import Entity, EntityType
from kg_search.extraction.relation_extractor import Relation, RelationType
from kg_search.ingestion.chunkers.base import Chunk
from kg_search.ingestion.loaders.base import Document
from kg_search.utils import generate_id, get_logger

from .config import CULTURAL_RELIC_ENTITY_TYPES, CULTURAL_RELIC_RELATION_TYPES, GraphRAGConfig

logger = get_logger(__name__)


# GraphRAG风格的实体抽取Prompt（中文文博领域优化）
GRAPHRAG_ENTITY_EXTRACTION_PROMPT = """你是一个专业的文博领域知识图谱构建专家。请从以下文本中识别和提取实体。

## 实体类型
请识别以下类型的实体：
{entity_types}

## 输入文本
{text}

## 抽取要求
1. 仔细识别文本中的所有实体
2. 对每个实体提供：名称、类型、描述
3. 实体描述应包含从文本中获取的关键信息
4. 如果同一实体出现多次，合并信息
5. 保持实体名称的规范化

## 输出格式
请以JSON格式输出，结构如下：
{{
    "entities": [
        {{
            "name": "实体名称",
            "type": "实体类型",
            "description": "实体描述（100字以内）"
        }}
    ]
}}

请开始抽取："""


GRAPHRAG_RELATION_EXTRACTION_PROMPT = """你是一个专业的文博领域知识图谱构建专家。请根据已识别的实体，从文本中提取实体间的关系。

## 已识别实体
{entities}

## 关系类型
请识别以下类型的关系：
{relation_types}

## 输入文本
{text}

## 抽取要求
1. 识别实体之间存在的关系
2. 只提取文本明确支持的关系
3. 对每个关系提供：源实体、目标实体、关系类型、描述、置信度
4. 置信度范围0-1，表示关系的确定程度

## 输出格式
请以JSON格式输出，结构如下：
{{
    "relations": [
        {{
            "source": "源实体名称",
            "target": "目标实体名称",
            "relation_type": "关系类型",
            "description": "关系描述",
            "weight": 1.0
        }}
    ]
}}

请开始抽取："""


class GraphRAGEntityExtractor:
    """
    GraphRAG风格的实体抽取器

    采用GraphRAG的抽取策略，适配自研数据结构
    """

    def __init__(
        self,
        llm_client: Any,
        config: GraphRAGConfig | None = None,
    ):
        """
        初始化实体抽取器

        Args:
            llm_client: LLM客户端
            config: GraphRAG配置
        """
        self.llm_client = llm_client
        self.config = config or GraphRAGConfig.from_settings()
        self.entity_types = CULTURAL_RELIC_ENTITY_TYPES

    async def extract_entities(
        self,
        text: str,
        source_doc_id: str = "",
        source_chunk_id: str = "",
    ) -> list[Entity]:
        """
        从文本中抽取实体

        Args:
            text: 输入文本
            source_doc_id: 来源文档ID
            source_chunk_id: 来源块ID

        Returns:
            实体列表
        """
        entity_types_str = "、".join(self.entity_types)

        prompt = GRAPHRAG_ENTITY_EXTRACTION_PROMPT.format(
            entity_types=entity_types_str,
            text=text[: self.config.entity_extract_max_tokens],
        )

        try:
            response = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            result = json.loads(response)
            entities_data = result.get("entities", [])

            entities = []
            for data in entities_data:
                entity = self._create_entity(
                    data,
                    source_doc_id=source_doc_id,
                    source_chunk_id=source_chunk_id,
                )
                entities.append(entity)

            logger.info(
                "GraphRAG entity extraction completed",
                count=len(entities),
                source_doc_id=source_doc_id,
            )
            return entities

        except Exception as e:
            logger.error("GraphRAG entity extraction failed", error=str(e))
            return []

    async def extract_from_document(self, document: Document) -> list[Entity]:
        """从文档中抽取实体"""
        # 首先从结构化字段创建实体
        entities = self._extract_structured_entities(document)

        # 然后从描述文本中用LLM提取
        if document.description:
            text_entities = await self.extract_entities(
                document.description,
                source_doc_id=document.id,
            )
            entities.extend(text_entities)

        return self._deduplicate_entities(entities)

    async def extract_from_chunk(self, chunk: Chunk) -> list[Entity]:
        """从文本块中抽取实体"""
        return await self.extract_entities(
            chunk.content,
            source_doc_id=chunk.document_id,
            source_chunk_id=chunk.id,
        )

    def _create_entity(
        self,
        data: dict[str, Any],
        source_doc_id: str = "",
        source_chunk_id: str = "",
    ) -> Entity:
        """创建实体对象"""
        name = data.get("name", "")
        type_str = data.get("type", "")

        # 尝试映射到EntityType枚举
        try:
            entity_type = EntityType(type_str)
        except ValueError:
            entity_type = type_str

        return Entity(
            id=generate_id(f"entity_{type_str}"),
            name=name,
            type=entity_type,
            description=data.get("description", ""),
            attributes=data.get("attributes", {}),
            source_doc_id=source_doc_id,
            source_chunk_id=source_chunk_id,
            confidence=data.get("confidence", 1.0),
        )

    def _extract_structured_entities(self, document: Document) -> list[Entity]:
        """从文档结构化字段提取实体"""
        entities = []

        # 文物名称
        if document.name:
            entities.append(
                Entity(
                    id=generate_id("entity_artifact"),
                    name=document.name,
                    type=EntityType.ARTIFACT,
                    description=document.description or "",
                    source_doc_id=document.id,
                )
            )

        # 朝代
        if dynasty := document.metadata.get("dynasty"):
            entities.append(
                Entity(
                    id=generate_id("entity_dynasty"),
                    name=dynasty,
                    type=EntityType.DYNASTY,
                    source_doc_id=document.id,
                )
            )

        # 材质
        if material := document.metadata.get("material"):
            entities.append(
                Entity(
                    id=generate_id("entity_material"),
                    name=material,
                    type=EntityType.MATERIAL,
                    source_doc_id=document.id,
                )
            )

        # 收藏机构
        if museum := document.metadata.get("museum"):
            entities.append(
                Entity(
                    id=generate_id("entity_museum"),
                    name=museum,
                    type=EntityType.MUSEUM,
                    source_doc_id=document.id,
                )
            )

        # 出土地点
        if location := document.metadata.get("excavation_site"):
            entities.append(
                Entity(
                    id=generate_id("entity_location"),
                    name=location,
                    type=EntityType.LOCATION,
                    source_doc_id=document.id,
                )
            )

        return entities

    def _deduplicate_entities(self, entities: list[Entity]) -> list[Entity]:
        """实体去重"""
        seen = {}
        unique = []

        for entity in entities:
            key = (entity.name.lower(), str(entity.type))
            if key not in seen:
                seen[key] = entity
                unique.append(entity)
            else:
                # 合并属性和描述
                existing = seen[key]
                existing.attributes.update(entity.attributes)
                if entity.description and not existing.description:
                    existing.description = entity.description

        return unique


class GraphRAGRelationExtractor:
    """
    GraphRAG风格的关系抽取器

    采用GraphRAG的抽取策略，适配自研数据结构
    """

    def __init__(
        self,
        llm_client: Any,
        config: GraphRAGConfig | None = None,
    ):
        """
        初始化关系抽取器

        Args:
            llm_client: LLM客户端
            config: GraphRAG配置
        """
        self.llm_client = llm_client
        self.config = config or GraphRAGConfig.from_settings()
        self.relation_types = CULTURAL_RELIC_RELATION_TYPES

    async def extract_relations(
        self,
        text: str,
        entities: list[Entity],
        source_doc_id: str = "",
    ) -> list[Relation]:
        """
        从文本中抽取关系

        Args:
            text: 输入文本
            entities: 已识别的实体列表
            source_doc_id: 来源文档ID

        Returns:
            关系列表
        """
        if len(entities) < 2:
            return []

        # 构建实体字符串
        entities_str = "\n".join(
            [
                f"- {e.name} ({e.type.value if hasattr(e.type, 'value') else e.type})"
                for e in entities
            ]
        )
        relation_types_str = "、".join(self.relation_types)

        prompt = GRAPHRAG_RELATION_EXTRACTION_PROMPT.format(
            entities=entities_str,
            relation_types=relation_types_str,
            text=text[: self.config.entity_extract_max_tokens],
        )

        try:
            response = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            result = json.loads(response)
            relations_data = result.get("relations", [])

            # 构建实体名称到ID的映射
            entity_map = {e.name: e.id for e in entities}

            relations = []
            for data in relations_data:
                relation = self._create_relation(
                    data,
                    entity_map=entity_map,
                    source_doc_id=source_doc_id,
                )
                if relation:
                    relations.append(relation)

            # 限制关系数量
            relations = relations[: self.config.max_relationships_per_chunk]

            logger.info(
                "GraphRAG relation extraction completed",
                count=len(relations),
                source_doc_id=source_doc_id,
            )
            return relations

        except Exception as e:
            logger.error("GraphRAG relation extraction failed", error=str(e))
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
            type_key = entity.type.value if hasattr(entity.type, "value") else str(entity.type)
            if type_key not in entities_by_type:
                entities_by_type[type_key] = []
            entities_by_type[type_key].append(entity)

        # 获取文物实体
        artifacts = entities_by_type.get("文物", [])

        # 为每个文物创建与其他实体的关系
        for artifact in artifacts:
            # 与朝代的关系
            for dynasty in entities_by_type.get("朝代", []):
                if artifact.source_doc_id == dynasty.source_doc_id:
                    relations.append(
                        Relation(
                            id=generate_id("relation"),
                            source_entity_id=artifact.id,
                            source_entity_name=artifact.name,
                            target_entity_id=dynasty.id,
                            target_entity_name=dynasty.name,
                            relation_type=RelationType.BELONGS_TO_DYNASTY,
                            source_doc_id=source_doc_id,
                        )
                    )

            # 与材质的关系
            for material in entities_by_type.get("材质", []):
                if artifact.source_doc_id == material.source_doc_id:
                    relations.append(
                        Relation(
                            id=generate_id("relation"),
                            source_entity_id=artifact.id,
                            source_entity_name=artifact.name,
                            target_entity_id=material.id,
                            target_entity_name=material.name,
                            relation_type=RelationType.MADE_OF,
                            source_doc_id=source_doc_id,
                        )
                    )

            # 与收藏机构的关系
            for museum in entities_by_type.get("收藏机构", []):
                if artifact.source_doc_id == museum.source_doc_id:
                    relations.append(
                        Relation(
                            id=generate_id("relation"),
                            source_entity_id=artifact.id,
                            source_entity_name=artifact.name,
                            target_entity_id=museum.id,
                            target_entity_name=museum.name,
                            relation_type=RelationType.COLLECTED_BY,
                            source_doc_id=source_doc_id,
                        )
                    )

            # 与出土地点的关系
            for location in entities_by_type.get("地点", []):
                if artifact.source_doc_id == location.source_doc_id:
                    relations.append(
                        Relation(
                            id=generate_id("relation"),
                            source_entity_id=artifact.id,
                            source_entity_name=artifact.name,
                            target_entity_id=location.id,
                            target_entity_name=location.name,
                            relation_type=RelationType.EXCAVATED_FROM,
                            source_doc_id=source_doc_id,
                        )
                    )

        return relations

    def _create_relation(
        self,
        data: dict[str, Any],
        entity_map: dict[str, str],
        source_doc_id: str = "",
    ) -> Relation | None:
        """创建关系对象"""
        source_name = data.get("source", "")
        target_name = data.get("target", "")

        # 检查实体是否存在
        source_id = entity_map.get(source_name, "")
        target_id = entity_map.get(target_name, "")

        if not source_name or not target_name:
            return None

        rel_type_str = data.get("relation_type", "相关")

        # 尝试映射到RelationType枚举
        try:
            relation_type = RelationType(rel_type_str)
        except ValueError:
            relation_type = rel_type_str

        return Relation(
            id=generate_id("relation"),
            source_entity_id=source_id,
            source_entity_name=source_name,
            target_entity_id=target_id,
            target_entity_name=target_name,
            relation_type=relation_type,
            description=data.get("description", ""),
            weight=data.get("weight", 1.0),
            confidence=data.get("confidence", 1.0),
            source_doc_id=source_doc_id,
        )
