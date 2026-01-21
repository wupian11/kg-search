"""
知识图谱构建器

整合实体和关系，构建知识图谱
"""

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from kg_search.ingestion.chunkers.base import Chunk
from kg_search.ingestion.loaders.base import Document
from kg_search.utils import get_logger

from .entity_extractor import Entity, EntityExtractor
from .relation_extractor import Relation, RelationExtractor

logger = get_logger(__name__)


@dataclass
class KnowledgeGraph:
    """知识图谱数据结构"""

    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    communities: list[dict[str, Any]] = field(default_factory=list)

    def to_networkx(self) -> nx.Graph:
        """转换为NetworkX图"""
        graph = nx.Graph()

        # 添加节点
        for entity in self.entities:
            graph.add_node(
                entity.id,
                name=entity.name,
                type=entity.type.value if hasattr(entity.type, "value") else entity.type,
                description=entity.description,
                **entity.attributes,
            )

        # 添加边
        for relation in self.relations:
            if relation.source_entity_id and relation.target_entity_id:
                graph.add_edge(
                    relation.source_entity_id,
                    relation.target_entity_id,
                    relation_type=relation.relation_type.value
                    if hasattr(relation.relation_type, "value")
                    else relation.relation_type,
                    description=relation.description,
                    weight=relation.weight,
                )

        return graph

    def get_entity_by_id(self, entity_id: str) -> Entity | None:
        """根据ID获取实体"""
        for entity in self.entities:
            if entity.id == entity_id:
                return entity
        return None

    def get_entity_by_name(self, name: str) -> Entity | None:
        """根据名称获取实体"""
        for entity in self.entities:
            if entity.name == name:
                return entity
        return None

    def get_relations_for_entity(self, entity_id: str) -> list[Relation]:
        """获取实体相关的所有关系"""
        return [
            r
            for r in self.relations
            if r.source_entity_id == entity_id or r.target_entity_id == entity_id
        ]

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "communities": self.communities,
        }


class GraphBuilder:
    """知识图谱构建器"""

    def __init__(self, llm_client: Any, use_graphrag: bool | None = None):
        """
        初始化图谱构建器

        Args:
            llm_client: LLM客户端实例
            use_graphrag: 是否使用GraphRAG风格抽取器（None表示从配置读取）
        """
        from kg_search.config import get_settings

        self.llm_client = llm_client

        settings = get_settings()
        self._use_graphrag = (
            use_graphrag if use_graphrag is not None else settings.use_graphrag_extractor
        )

        if self._use_graphrag:
            # 使用GraphRAG风格抽取器
            from kg_search.graphrag import GraphRAGEntityExtractor, GraphRAGRelationExtractor

            self.entity_extractor = GraphRAGEntityExtractor(llm_client)
            self.relation_extractor = GraphRAGRelationExtractor(llm_client)
            logger.info("GraphBuilder using GraphRAG extractors")
        else:
            # 使用原生抽取器
            self.entity_extractor = EntityExtractor(llm_client)
            self.relation_extractor = RelationExtractor(llm_client)
            logger.info("GraphBuilder using native extractors")

    async def build_from_documents(
        self,
        documents: list[Document],
        extract_from_text: bool = True,
    ) -> KnowledgeGraph:
        """
        从文档列表构建知识图谱

        Args:
            documents: 文档列表
            extract_from_text: 是否从文本中用LLM提取

        Returns:
            知识图谱
        """
        all_entities: list[Entity] = []
        all_relations: list[Relation] = []

        for doc in documents:
            logger.info("Processing document", doc_id=doc.id)

            # 提取实体
            entities = await self.entity_extractor.extract_from_document(doc)
            all_entities.extend(entities)

            # 提取结构化关系
            relations = self.relation_extractor.extract_structured_relations(
                entities, source_doc_id=doc.id
            )
            all_relations.extend(relations)

            # 从文本中提取额外关系
            if extract_from_text and doc.description:
                text_relations = await self.relation_extractor.extract_from_text(
                    doc.description, entities, source_doc_id=doc.id
                )
                all_relations.extend(text_relations)

        # 去重
        entities = self._deduplicate_entities(all_entities)
        relations = self._deduplicate_relations(all_relations)

        logger.info(
            "Graph built from documents",
            entity_count=len(entities),
            relation_count=len(relations),
        )

        return KnowledgeGraph(entities=entities, relations=relations)

    async def build_from_chunks(
        self,
        chunks: list[Chunk],
    ) -> KnowledgeGraph:
        """
        从文本块列表构建知识图谱

        Args:
            chunks: 文本块列表

        Returns:
            知识图谱
        """
        all_entities: list[Entity] = []
        all_relations: list[Relation] = []

        for chunk in chunks:
            logger.info("Processing chunk", chunk_id=chunk.id)

            # 提取实体
            entities = await self.entity_extractor.extract_from_chunk(chunk)
            all_entities.extend(entities)

            # 提取关系
            if len(entities) >= 2:
                relations = await self.relation_extractor.extract_from_text(
                    chunk.content, entities, source_doc_id=chunk.document_id
                )
                all_relations.extend(relations)

        # 去重
        entities = self._deduplicate_entities(all_entities)
        relations = self._deduplicate_relations(all_relations)

        logger.info(
            "Graph built from chunks",
            entity_count=len(entities),
            relation_count=len(relations),
        )

        return KnowledgeGraph(entities=entities, relations=relations)

    def merge_graphs(self, graphs: list[KnowledgeGraph]) -> KnowledgeGraph:
        """
        合并多个知识图谱

        Args:
            graphs: 知识图谱列表

        Returns:
            合并后的知识图谱
        """
        all_entities = []
        all_relations = []

        for graph in graphs:
            all_entities.extend(graph.entities)
            all_relations.extend(graph.relations)

        entities = self._deduplicate_entities(all_entities)
        relations = self._deduplicate_relations(all_relations)

        return KnowledgeGraph(entities=entities, relations=relations)

    def _deduplicate_entities(self, entities: list[Entity]) -> list[Entity]:
        """实体去重"""
        seen = {}
        unique = []

        for entity in entities:
            key = (entity.name.lower(), entity.type)
            if key not in seen:
                seen[key] = entity
                unique.append(entity)
            else:
                # 合并属性
                existing = seen[key]
                existing.attributes.update(entity.attributes)
                if entity.description and not existing.description:
                    existing.description = entity.description

        return unique

    def _deduplicate_relations(self, relations: list[Relation]) -> list[Relation]:
        """关系去重"""
        seen = set()
        unique = []

        for relation in relations:
            key = (
                relation.source_entity_name.lower(),
                relation.target_entity_name.lower(),
                relation.relation_type,
            )
            if key not in seen:
                seen.add(key)
                unique.append(relation)

        return unique
