"""
图检索器

基于知识图谱的关联检索
"""

from typing import Any

from kg_search.extraction.entity_extractor import Entity
from kg_search.indexing.graph_store import GraphStore
from kg_search.utils import get_logger

from .vector_retriever import RetrievalResult

logger = get_logger(__name__)


class GraphRetriever:
    """图检索器"""

    def __init__(self, graph_store: GraphStore):
        """
        初始化图检索器

        Args:
            graph_store: 图存储
        """
        self.graph_store = graph_store

    async def retrieve_by_entity(
        self,
        entity_name: str,
        relation_types: list[str] | None = None,
        depth: int = 2,
        limit: int = 20,
    ) -> list[RetrievalResult]:
        """
        基于实体的关联检索

        Args:
            entity_name: 实体名称
            relation_types: 关系类型过滤
            depth: 搜索深度
            limit: 返回数量限制

        Returns:
            检索结果列表
        """
        # 搜索实体
        entities = await self.graph_store.search_entities(entity_name, limit=5)

        if not entities:
            logger.info("No entities found", entity_name=entity_name)
            return []

        # 获取第一个匹配实体的邻居
        entity = entities[0]
        neighbors = await self.graph_store.get_neighbors(
            entity.id,
            relation_types=relation_types,
            depth=depth,
        )

        # 构建结果
        results = []

        # 添加主实体
        results.append(
            RetrievalResult(
                id=entity.id,
                content=self._entity_to_content(entity),
                score=1.0,
                metadata={
                    "type": "entity",
                    "entity_type": entity.type.value
                    if hasattr(entity.type, "value")
                    else entity.type,
                    "entity_name": entity.name,
                },
                source_type="graph",
            )
        )

        # 添加邻居实体
        for neighbor in neighbors[:limit]:
            neighbor_entity = neighbor["entity"]
            # 距离越近分数越高
            score = 1.0 / (neighbor["distance"] + 1)

            results.append(
                RetrievalResult(
                    id=neighbor_entity["id"],
                    content=self._neighbor_to_content(neighbor),
                    score=score,
                    metadata={
                        "type": "entity",
                        "entity_type": neighbor_entity["type"],
                        "entity_name": neighbor_entity["name"],
                        "relations": neighbor["relation_types"],
                        "distance": neighbor["distance"],
                    },
                    source_type="graph",
                )
            )

        logger.info(
            "Graph retrieval completed",
            entity_name=entity_name,
            results_count=len(results),
        )

        return results

    async def retrieve_by_query(
        self,
        query: str,
        entity_types: list[str] | None = None,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        """
        基于关键词的图检索

        Args:
            query: 查询文本
            entity_types: 实体类型过滤
            limit: 返回数量限制

        Returns:
            检索结果列表
        """
        # 搜索匹配的实体
        entities = await self.graph_store.search_entities(
            query,
            entity_types=entity_types,
            limit=limit,
        )

        results = []
        for i, entity in enumerate(entities):
            # 基于排名计算分数
            score = 1.0 / (i + 1)

            results.append(
                RetrievalResult(
                    id=entity.id,
                    content=self._entity_to_content(entity),
                    score=score,
                    metadata={
                        "type": "entity",
                        "entity_type": entity.type.value
                        if hasattr(entity.type, "value")
                        else entity.type,
                        "entity_name": entity.name,
                    },
                    source_type="graph",
                )
            )

        return results

    async def retrieve_path(
        self,
        start_entity: str,
        end_entity: str,
        max_depth: int = 4,
    ) -> list[RetrievalResult]:
        """
        查找两个实体之间的路径

        Args:
            start_entity: 起始实体名称
            end_entity: 目标实体名称
            max_depth: 最大路径长度

        Returns:
            路径上的实体和关系
        """
        # 搜索起始和目标实体
        start_entities = await self.graph_store.search_entities(start_entity, limit=1)
        end_entities = await self.graph_store.search_entities(end_entity, limit=1)

        if not start_entities or not end_entities:
            return []

        # TODO: 实现路径查找（需要扩展GraphStore接口）
        # 目前返回两个实体的信息
        results = []

        for entity in [start_entities[0], end_entities[0]]:
            results.append(
                RetrievalResult(
                    id=entity.id,
                    content=self._entity_to_content(entity),
                    score=1.0,
                    metadata={
                        "type": "entity",
                        "entity_type": entity.type.value
                        if hasattr(entity.type, "value")
                        else entity.type,
                        "entity_name": entity.name,
                    },
                    source_type="graph",
                )
            )

        return results

    async def retrieve_by_relation(
        self,
        relation_type: str,
        limit: int = 20,
    ) -> list[RetrievalResult]:
        """
        按关系类型检索

        Args:
            relation_type: 关系类型
            limit: 返回数量限制

        Returns:
            具有该关系的实体对
        """
        # TODO: 实现基于关系类型的检索
        return []

    def _entity_to_content(self, entity: Entity) -> str:
        """将实体转换为文本内容"""
        entity_type = entity.type.value if hasattr(entity.type, "value") else entity.type
        parts = [
            f"{entity_type}：{entity.name}",
        ]
        if entity.description:
            parts.append(f"描述：{entity.description}")
        return "\n".join(parts)

    def _neighbor_to_content(self, neighbor: dict[str, Any]) -> str:
        """将邻居信息转换为文本内容"""
        entity = neighbor["entity"]
        relations = neighbor.get("relation_types", [])

        parts = [
            f"{entity['type']}：{entity['name']}",
        ]
        if entity.get("description"):
            parts.append(f"描述：{entity['description']}")
        if relations:
            parts.append(f"关系：{', '.join(relations)}")

        return "\n".join(parts)
