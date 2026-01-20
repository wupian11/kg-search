"""
图数据库存储

使用Neo4j存储知识图谱
"""

from abc import ABC, abstractmethod
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

from kg_search.config import get_settings
from kg_search.extraction.entity_extractor import Entity
from kg_search.extraction.graph_builder import KnowledgeGraph
from kg_search.extraction.relation_extractor import Relation
from kg_search.utils import get_logger

logger = get_logger(__name__)


class GraphStore(ABC):
    """图存储抽象基类"""

    @abstractmethod
    async def add_entities(self, entities: list[Entity]) -> None:
        """添加实体"""
        pass

    @abstractmethod
    async def add_relations(self, relations: list[Relation]) -> None:
        """添加关系"""
        pass

    @abstractmethod
    async def get_entity(self, entity_id: str) -> Entity | None:
        """获取实体"""
        pass

    @abstractmethod
    async def get_neighbors(
        self,
        entity_id: str,
        relation_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[dict[str, Any]]:
        """获取邻居节点"""
        pass

    @abstractmethod
    async def search_entities(
        self,
        query: str,
        entity_types: list[str] | None = None,
        limit: int = 10,
    ) -> list[Entity]:
        """搜索实体"""
        pass


class Neo4jGraphStore(GraphStore):
    """Neo4j图存储"""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        """
        初始化Neo4j连接

        Args:
            uri: Neo4j URI
            user: 用户名
            password: 密码
            database: 数据库名
        """
        settings = get_settings()

        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self.database = database or settings.neo4j_database

        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """建立连接"""
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            logger.info("Connected to Neo4j", uri=self.uri)

    async def close(self) -> None:
        """关闭连接"""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("Disconnected from Neo4j")

    async def _get_session(self):
        """获取数据库会话"""
        if self._driver is None:
            await self.connect()
        return self._driver.session(database=self.database)

    async def init_schema(self) -> None:
        """初始化数据库schema（创建索引和约束）"""
        async with await self._get_session() as session:
            # 创建实体ID唯一约束
            await session.run(
                "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE"
            )

            # 创建实体名称索引
            await session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")

            # 创建实体类型索引
            await session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)")

            # 创建全文搜索索引
            await session.run("""
                CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
                FOR (e:Entity) ON EACH [e.name, e.description]
            """)

            logger.info("Neo4j schema initialized")

    async def add_entities(self, entities: list[Entity]) -> None:
        """添加实体到Neo4j"""
        if not entities:
            return

        async with await self._get_session() as session:
            for entity in entities:
                entity_type = entity.type.value if hasattr(entity.type, "value") else entity.type

                query = """
                MERGE (e:Entity {id: $id})
                SET e.name = $name,
                    e.type = $type,
                    e.description = $description,
                    e.source_doc_id = $source_doc_id,
                    e.attributes = $attributes
                WITH e
                CALL apoc.create.addLabels(e, [$type_label]) YIELD node
                RETURN node
                """

                try:
                    await session.run(
                        query,
                        id=entity.id,
                        name=entity.name,
                        type=entity_type,
                        description=entity.description,
                        source_doc_id=entity.source_doc_id,
                        attributes=str(entity.attributes),
                        type_label=entity_type.replace(" ", "_"),
                    )
                except Exception:
                    # 如果APOC不可用，使用简单查询
                    simple_query = """
                    MERGE (e:Entity {id: $id})
                    SET e.name = $name,
                        e.type = $type,
                        e.description = $description,
                        e.source_doc_id = $source_doc_id
                    """
                    await session.run(
                        simple_query,
                        id=entity.id,
                        name=entity.name,
                        type=entity_type,
                        description=entity.description,
                        source_doc_id=entity.source_doc_id,
                    )

        logger.info("Added entities to Neo4j", count=len(entities))

    async def add_relations(self, relations: list[Relation]) -> None:
        """添加关系到Neo4j"""
        if not relations:
            return

        async with await self._get_session() as session:
            for relation in relations:
                rel_type = (
                    relation.relation_type.value
                    if hasattr(relation.relation_type, "value")
                    else relation.relation_type
                )
                # 将关系类型转换为有效的Neo4j关系类型名称
                rel_type_clean = rel_type.replace(" ", "_").upper()

                query = f"""
                MATCH (source:Entity {{id: $source_id}})
                MATCH (target:Entity {{id: $target_id}})
                MERGE (source)-[r:{rel_type_clean}]->(target)
                SET r.id = $rel_id,
                    r.description = $description,
                    r.weight = $weight,
                    r.source_doc_id = $source_doc_id
                """

                await session.run(
                    query,
                    source_id=relation.source_entity_id,
                    target_id=relation.target_entity_id,
                    rel_id=relation.id,
                    description=relation.description,
                    weight=relation.weight,
                    source_doc_id=relation.source_doc_id,
                )

        logger.info("Added relations to Neo4j", count=len(relations))

    async def add_knowledge_graph(self, kg: KnowledgeGraph) -> None:
        """添加完整知识图谱"""
        await self.add_entities(kg.entities)
        await self.add_relations(kg.relations)

    async def get_entity(self, entity_id: str) -> Entity | None:
        """获取实体"""
        async with await self._get_session() as session:
            result = await session.run(
                "MATCH (e:Entity {id: $id}) RETURN e",
                id=entity_id,
            )
            record = await result.single()

            if record:
                node = record["e"]
                return Entity(
                    id=node["id"],
                    name=node["name"],
                    type=node["type"],
                    description=node.get("description", ""),
                    source_doc_id=node.get("source_doc_id", ""),
                )
            return None

    async def get_neighbors(
        self,
        entity_id: str,
        relation_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[dict[str, Any]]:
        """
        获取邻居节点

        Args:
            entity_id: 实体ID
            relation_types: 关系类型过滤
            depth: 搜索深度

        Returns:
            邻居节点和关系信息
        """
        async with await self._get_session() as session:
            # 构建关系类型过滤
            rel_filter = ""
            if relation_types:
                rel_types = [rt.replace(" ", "_").upper() for rt in relation_types]
                rel_filter = ":" + "|".join(rel_types)

            query = f"""
            MATCH (start:Entity {{id: $id}})-[r{rel_filter}*1..{depth}]-(neighbor:Entity)
            RETURN DISTINCT neighbor,
                   [rel in r | type(rel)] as relation_types,
                   length(r) as distance
            ORDER BY distance
            LIMIT 50
            """

            result = await session.run(query, id=entity_id)
            records = await result.data()

            neighbors = []
            for record in records:
                node = record["neighbor"]
                neighbors.append(
                    {
                        "entity": {
                            "id": node["id"],
                            "name": node["name"],
                            "type": node["type"],
                            "description": node.get("description", ""),
                        },
                        "relation_types": record["relation_types"],
                        "distance": record["distance"],
                    }
                )

            return neighbors

    async def search_entities(
        self,
        query: str,
        entity_types: list[str] | None = None,
        limit: int = 10,
    ) -> list[Entity]:
        """
        搜索实体

        Args:
            query: 搜索关键词
            entity_types: 实体类型过滤
            limit: 返回数量限制

        Returns:
            实体列表
        """
        async with await self._get_session() as session:
            # 尝试使用全文搜索
            try:
                cypher = """
                CALL db.index.fulltext.queryNodes('entity_fulltext', $query)
                YIELD node, score
                WHERE ($types IS NULL OR node.type IN $types)
                RETURN node, score
                ORDER BY score DESC
                LIMIT $limit
                """
                result = await session.run(
                    cypher,
                    query=query,
                    types=entity_types,
                    limit=limit,
                )
            except Exception:
                # 回退到模糊匹配
                cypher = """
                MATCH (e:Entity)
                WHERE e.name CONTAINS $query OR e.description CONTAINS $query
                AND ($types IS NULL OR e.type IN $types)
                RETURN e as node
                LIMIT $limit
                """
                result = await session.run(
                    cypher,
                    query=query,
                    types=entity_types,
                    limit=limit,
                )

            records = await result.data()

            entities = []
            for record in records:
                node = record["node"]
                entities.append(
                    Entity(
                        id=node["id"],
                        name=node["name"],
                        type=node["type"],
                        description=node.get("description", ""),
                    )
                )

            return entities

    async def get_subgraph(
        self,
        entity_ids: list[str],
        depth: int = 1,
    ) -> KnowledgeGraph:
        """
        获取子图

        Args:
            entity_ids: 起始实体ID列表
            depth: 扩展深度

        Returns:
            子图
        """
        async with await self._get_session() as session:
            query = f"""
            MATCH (start:Entity)
            WHERE start.id IN $ids
            CALL apoc.path.subgraphAll(start, {{maxLevel: {depth}}})
            YIELD nodes, relationships
            RETURN nodes, relationships
            """

            try:
                result = await session.run(query, ids=entity_ids)
                record = await result.single()

                if record:
                    entities = []
                    for node in record["nodes"]:
                        entities.append(
                            Entity(
                                id=node["id"],
                                name=node["name"],
                                type=node["type"],
                                description=node.get("description", ""),
                            )
                        )

                    relations = []
                    for rel in record["relationships"]:
                        relations.append(
                            Relation(
                                id=rel.get("id", ""),
                                source_entity_id=rel.start_node["id"],
                                source_entity_name=rel.start_node["name"],
                                target_entity_id=rel.end_node["id"],
                                target_entity_name=rel.end_node["name"],
                                relation_type=rel.type,
                            )
                        )

                    return KnowledgeGraph(entities=entities, relations=relations)
            except Exception:
                # APOC不可用时的简单实现
                pass

            return KnowledgeGraph()

    async def clear(self) -> None:
        """清空数据库"""
        async with await self._get_session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
            logger.warning("Neo4j database cleared")
