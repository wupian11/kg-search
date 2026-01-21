"""
GraphRAG主适配器

整合所有GraphRAG功能，提供统一接口
"""

from typing import Any

from kg_search.extraction.graph_builder import KnowledgeGraph
from kg_search.indexing.graph_store import GraphStore
from kg_search.ingestion.chunkers.base import Chunk
from kg_search.ingestion.loaders.base import Document
from kg_search.retrieval.vector_retriever import VectorRetriever
from kg_search.utils import get_logger

from .config import GraphRAGConfig
from .extractors import GraphRAGEntityExtractor, GraphRAGRelationExtractor
from .searchers import CommunityReport, GraphRAGGlobalSearcher, GraphRAGLocalSearcher

logger = get_logger(__name__)


class GraphRAGAdapter:
    """
    GraphRAG适配器

    整合GraphRAG的索引和查询能力，与自研系统无缝对接
    """

    def __init__(
        self,
        llm_client: Any,
        vector_retriever: VectorRetriever | None = None,
        graph_store: GraphStore | None = None,
        config: GraphRAGConfig | None = None,
    ):
        """
        初始化GraphRAG适配器

        Args:
            llm_client: LLM客户端
            vector_retriever: 向量检索器（Local Search需要）
            graph_store: 图存储（Local Search需要）
            config: GraphRAG配置
        """
        self.llm_client = llm_client
        self.vector_retriever = vector_retriever
        self.graph_store = graph_store
        self.config = config or GraphRAGConfig.from_settings()

        # 初始化抽取器
        self.entity_extractor = GraphRAGEntityExtractor(llm_client, self.config)
        self.relation_extractor = GraphRAGRelationExtractor(llm_client, self.config)

        # 初始化搜索器
        self.global_searcher = GraphRAGGlobalSearcher(llm_client, self.config)

        if vector_retriever and graph_store:
            self.local_searcher = GraphRAGLocalSearcher(
                vector_retriever,
                graph_store,
                llm_client,
                self.config,
            )
        else:
            self.local_searcher = None

        logger.info("GraphRAG adapter initialized")

    # ==================== 索引构建 ====================

    async def build_graph_from_documents(
        self,
        documents: list[Document],
        extract_from_text: bool = True,
    ) -> KnowledgeGraph:
        """
        从文档构建知识图谱

        Args:
            documents: 文档列表
            extract_from_text: 是否从文本中提取实体/关系

        Returns:
            知识图谱
        """
        all_entities = []
        all_relations = []

        for doc in documents:
            logger.info("Processing document with GraphRAG", doc_id=doc.id)

            # 抽取实体
            entities = await self.entity_extractor.extract_from_document(doc)
            all_entities.extend(entities)

            # 抽取结构化关系
            relations = self.relation_extractor.extract_structured_relations(
                entities,
                source_doc_id=doc.id,
            )
            all_relations.extend(relations)

            # 从文本中抽取关系
            if extract_from_text and doc.description:
                text_relations = await self.relation_extractor.extract_relations(
                    doc.description,
                    entities,
                    source_doc_id=doc.id,
                )
                all_relations.extend(text_relations)

        # 去重
        entities = self._deduplicate_entities(all_entities)
        relations = self._deduplicate_relations(all_relations)

        logger.info(
            "GraphRAG graph built",
            entity_count=len(entities),
            relation_count=len(relations),
        )

        return KnowledgeGraph(entities=entities, relations=relations)

    async def build_graph_from_chunks(
        self,
        chunks: list[Chunk],
    ) -> KnowledgeGraph:
        """
        从文本块构建知识图谱

        Args:
            chunks: 文本块列表

        Returns:
            知识图谱
        """
        all_entities = []
        all_relations = []

        for chunk in chunks:
            logger.info("Processing chunk with GraphRAG", chunk_id=chunk.id)

            # 抽取实体
            entities = await self.entity_extractor.extract_from_chunk(chunk)
            all_entities.extend(entities)

            # 抽取关系
            if len(entities) >= 2:
                relations = await self.relation_extractor.extract_relations(
                    chunk.content,
                    entities,
                    source_doc_id=chunk.document_id,
                )
                all_relations.extend(relations)

        # 去重
        entities = self._deduplicate_entities(all_entities)
        relations = self._deduplicate_relations(all_relations)

        return KnowledgeGraph(entities=entities, relations=relations)

    async def build_community_reports(
        self,
        knowledge_graph: KnowledgeGraph,
    ) -> list[CommunityReport]:
        """
        构建社区报告

        Args:
            knowledge_graph: 知识图谱

        Returns:
            社区报告列表
        """
        return await self.global_searcher.build_community_reports(knowledge_graph)

    # ==================== 查询接口 ====================

    async def global_search(
        self,
        query: str,
        community_level: int | None = None,
        response_type: str = "多段落",
    ) -> dict[str, Any]:
        """
        执行Global Search

        适用于宏观问题，如：
        - "中国青铜器的发展历程"
        - "商周时期的主要文物类型"

        Args:
            query: 用户问题
            community_level: 社区层级
            response_type: 响应类型

        Returns:
            搜索结果
        """
        return await self.global_searcher.search(
            query,
            community_level=community_level,
            response_type=response_type,
        )

    async def local_search(
        self,
        query: str,
        top_k: int = 10,
        include_neighbors: bool = True,
        neighbor_depth: int = 1,
    ) -> dict[str, Any]:
        """
        执行Local Search

        适用于具体问题，如：
        - "四羊方尊的材质是什么"
        - "这件文物是什么朝代的"

        Args:
            query: 用户问题
            top_k: 返回数量
            include_neighbors: 是否包含邻居
            neighbor_depth: 邻居深度

        Returns:
            搜索结果
        """
        if not self.local_searcher:
            return {
                "query": query,
                "answer": "Local Search未初始化，请提供vector_retriever和graph_store",
                "entities": [],
                "chunks": [],
            }

        return await self.local_searcher.search(
            query,
            top_k=top_k,
            include_neighbors=include_neighbors,
            neighbor_depth=neighbor_depth,
        )

    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        use_global: bool = True,
        use_local: bool = True,
    ) -> dict[str, Any]:
        """
        混合搜索（结合Global和Local）

        自动判断问题类型，选择合适的搜索策略

        Args:
            query: 用户问题
            top_k: 返回数量
            use_global: 是否使用Global Search
            use_local: 是否使用Local Search

        Returns:
            搜索结果
        """
        results = {
            "query": query,
            "global_result": None,
            "local_result": None,
            "combined_answer": "",
        }

        # 判断问题类型
        query_type = await self._classify_query(query)

        if use_global and query_type in ["global", "hybrid"]:
            results["global_result"] = await self.global_search(query)

        if use_local and query_type in ["local", "hybrid"]:
            results["local_result"] = await self.local_search(query, top_k=top_k)

        # 整合回答
        results["combined_answer"] = self._combine_answers(
            results["global_result"],
            results["local_result"],
        )

        return results

    async def _classify_query(self, query: str) -> str:
        """
        分类查询类型

        Returns:
            "global" | "local" | "hybrid"
        """
        prompt = f"""判断以下问题的类型：

问题：{query}

类型说明：
- global: 宏观问题，需要整体概述（如"青铜器的发展历程"）
- local: 具体问题，关于特定实体（如"四羊方尊是什么材质"）
- hybrid: 两者兼有

请只回答一个词：global、local 或 hybrid"""

        try:
            result = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )
            result = result.strip().lower()
            if result in ["global", "local", "hybrid"]:
                return result
        except Exception as e:
            logger.error("Query classification failed", error=str(e))

        return "hybrid"

    def _combine_answers(
        self,
        global_result: dict | None,
        local_result: dict | None,
    ) -> str:
        """整合Global和Local结果"""
        parts = []

        if local_result and local_result.get("answer"):
            parts.append(f"### 具体信息\n\n{local_result['answer']}")

        if global_result and global_result.get("answer"):
            parts.append(f"### 背景知识\n\n{global_result['answer']}")

        if not parts:
            return "抱歉，未能找到相关信息。"

        return "\n\n".join(parts)

    # ==================== 辅助方法 ====================

    def _deduplicate_entities(self, entities: list) -> list:
        """实体去重"""
        seen = {}
        unique = []

        for entity in entities:
            key = (entity.name.lower(), str(entity.type))
            if key not in seen:
                seen[key] = entity
                unique.append(entity)
            else:
                existing = seen[key]
                existing.attributes.update(entity.attributes)
                if entity.description and not existing.description:
                    existing.description = entity.description

        return unique

    def _deduplicate_relations(self, relations: list) -> list:
        """关系去重"""
        seen = set()
        unique = []

        for relation in relations:
            key = (
                relation.source_entity_name.lower(),
                relation.target_entity_name.lower(),
                str(relation.relation_type),
            )
            if key not in seen:
                seen.add(key)
                unique.append(relation)

        return unique

    def set_community_reports(self, reports: list[CommunityReport]) -> None:
        """设置预构建的社区报告"""
        self.global_searcher.community_reports = reports

    def get_community_reports(self) -> list[CommunityReport]:
        """获取当前社区报告"""
        return self.global_searcher.community_reports
