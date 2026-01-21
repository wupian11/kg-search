"""
GraphRAG检索器适配

封装GraphRAG的Global Search和Local Search功能
"""

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from kg_search.extraction.entity_extractor import Entity
from kg_search.extraction.graph_builder import KnowledgeGraph
from kg_search.indexing.graph_store import GraphStore
from kg_search.retrieval.vector_retriever import RetrievalResult, VectorRetriever
from kg_search.utils import get_logger

from .config import GraphRAGConfig

logger = get_logger(__name__)


@dataclass
class CommunityReport:
    """社区报告数据结构（GraphRAG标准格式）"""

    id: str
    community_id: str
    level: int
    title: str
    summary: str
    full_content: str = ""
    rank: float = 0.0
    entity_ids: list[str] = field(default_factory=list)
    relationship_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "community_id": self.community_id,
            "level": self.level,
            "title": self.title,
            "summary": self.summary,
            "full_content": self.full_content,
            "rank": self.rank,
            "entity_ids": self.entity_ids,
            "relationship_ids": self.relationship_ids,
        }


# GraphRAG Global Search的Map-Reduce Prompt
GLOBAL_SEARCH_MAP_PROMPT = """---角色---
你是一个专业的文博领域知识助手，负责回答关于文物、历史和文化遗产的问题。

---目标---
根据提供的社区报告数据生成回应，回答用户问题。报告按重要性排序。

如果你不知道答案，直接说不知道。不要编造任何内容。

用户问题可能涉及以下主题：
- 某类文物的特征和历史
- 某个朝代的文化特点
- 文物的材质和工艺
- 文物的收藏和出土情况

每个报告包含社区信息及其相关实体和关系的概述。

---数据报告---
{context_data}

---目标回应长度和格式---
{response_type}

根据回应长度和格式要求，适当添加章节和评论。使用markdown格式回应。
"""

GLOBAL_SEARCH_REDUCE_PROMPT = """---角色---
你是一个专业的文博领域知识助手，负责整合多个分析师的回答。

---目标---
给定一组关于用户问题的社区分析报告，整合这些报告生成最终的综合回答。

最终回答应该：
1. 移除所有无关信息
2. 合并相似的要点
3. 保留最重要和最相关的信息
4. 解决各报告之间的任何矛盾
5. 确保答案完整且连贯

如果所有报告都不包含有用信息，直接说不知道。不要编造内容。

---用户问题---
{query}

---分析报告---
{report_data}

---目标回应长度和格式---
{response_type}

使用markdown格式生成最终回答。
"""


class GraphRAGGlobalSearcher:
    """
    GraphRAG Global Search实现

    基于社区层级的Map-Reduce搜索策略
    适用于宏观问题，如"商周青铜器的发展特点"
    """

    def __init__(
        self,
        llm_client: Any,
        config: GraphRAGConfig | None = None,
    ):
        """
        初始化Global Searcher

        Args:
            llm_client: LLM客户端
            config: GraphRAG配置
        """
        self.llm_client = llm_client
        self.config = config or GraphRAGConfig.from_settings()
        self.community_reports: list[CommunityReport] = []

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
        graph = knowledge_graph.to_networkx()

        if graph.number_of_nodes() == 0:
            logger.warning("Empty knowledge graph")
            return []

        # 执行社区检测
        communities = self._detect_communities(graph)

        # 为每个社区生成报告
        reports = []
        for i, (community_nodes, level) in enumerate(communities):
            if len(community_nodes) > self.config.max_community_size:
                continue

            # 收集社区实体信息
            entity_infos = []
            for node_id in community_nodes:
                if graph.has_node(node_id):
                    node_data = graph.nodes[node_id]
                    entity_infos.append(
                        {
                            "id": node_id,
                            "name": node_data.get("name", node_id),
                            "type": node_data.get("type", ""),
                            "description": node_data.get("description", ""),
                        }
                    )

            # 收集社区内关系
            relationships = []
            for source, target, data in graph.edges(data=True):
                if source in community_nodes and target in community_nodes:
                    relationships.append(
                        {
                            "source": graph.nodes[source].get("name", source),
                            "target": graph.nodes[target].get("name", target),
                            "type": data.get("relation_type", "相关"),
                        }
                    )

            # 生成社区摘要和标题
            summary = await self._generate_community_summary(entity_infos, relationships)
            title = await self._generate_community_title(entity_infos, summary)
            rank = self._calculate_community_rank(entity_infos, relationships)

            report = CommunityReport(
                id=f"report_{i}",
                community_id=f"community_{i}",
                level=level,
                title=title,
                summary=summary,
                full_content=self._build_full_content(entity_infos, relationships),
                rank=rank,
                entity_ids=[e["id"] for e in entity_infos],
            )
            reports.append(report)

        # 按rank排序
        reports.sort(key=lambda x: x.rank, reverse=True)
        self.community_reports = reports

        logger.info("Community reports built", count=len(reports))
        return reports

    def _detect_communities(self, graph: nx.Graph) -> list[tuple[set[str], int]]:
        """执行社区检测"""
        if graph.number_of_nodes() < 2:
            return [(set(graph.nodes()), 0)]

        communities = []

        try:
            # 尝试使用Leiden算法
            from graspologic.partition import hierarchical_leiden

            node_list = list(graph.nodes())
            adj_matrix = nx.to_numpy_array(graph, nodelist=node_list)

            partition = hierarchical_leiden(
                adj_matrix,
                max_cluster_size=self.config.max_community_size,
            )

            for level, level_partition in enumerate(partition):
                for community_idx in set(level_partition):
                    community_nodes = {
                        node_list[i] for i, c in enumerate(level_partition) if c == community_idx
                    }
                    if len(community_nodes) > 1:
                        communities.append((community_nodes, level))

        except ImportError:
            logger.warning("graspologic not available, using Louvain")
            try:
                from networkx.algorithms.community import louvain_communities

                louvain_result = louvain_communities(graph)
                for i, community_nodes in enumerate(louvain_result):
                    communities.append((community_nodes, 0))

            except Exception as e:
                logger.error("Community detection failed", error=str(e))
                communities.append((set(graph.nodes()), 0))

        return communities

    async def _generate_community_summary(
        self,
        entity_infos: list[dict],
        relationships: list[dict],
    ) -> str:
        """生成社区摘要"""
        if not entity_infos:
            return ""

        entities_text = "\n".join(
            [
                f"- {e['name']} ({e['type']}): {e.get('description', '')[:100]}"
                for e in entity_infos[:15]
            ]
        )

        relations_text = "\n".join(
            [f"- {r['source']} --[{r['type']}]--> {r['target']}" for r in relationships[:10]]
        )

        prompt = f"""请为以下文博知识社区生成综合摘要。

## 社区实体
{entities_text}

## 实体关系
{relations_text}

## 要求
1. 概括社区的主题和共同特征
2. 描述实体之间的主要关系
3. 突出文化和历史价值
4. 300字以内

请生成摘要："""

        try:
            summary = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )
            return summary.strip()
        except Exception as e:
            logger.error("Summary generation failed", error=str(e))
            names = [e["name"] for e in entity_infos[:5]]
            return f"该社区包含：{', '.join(names)}等实体。"

    async def _generate_community_title(
        self,
        entity_infos: list[dict],
        summary: str,
    ) -> str:
        """生成社区标题"""
        prompt = f"""根据以下社区摘要，生成一个简短的标题（10-20字）。

摘要：{summary[:300]}

标题："""

        try:
            title = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )
            return title.strip()
        except Exception as e:
            logger.error("Title generation failed", error=str(e))
            if entity_infos:
                return f"包含{entity_infos[0]['name']}的社区"
            return "未命名社区"

    def _calculate_community_rank(
        self,
        entity_infos: list[dict],
        relationships: list[dict],
    ) -> float:
        """计算社区重要性排名"""
        entity_score = len(entity_infos) * 0.5
        relation_score = len(relationships) * 0.3

        # 文物数量加分
        artifact_count = sum(1 for e in entity_infos if e.get("type") == "文物")
        artifact_score = artifact_count * 0.2

        return entity_score + relation_score + artifact_score

    def _build_full_content(
        self,
        entity_infos: list[dict],
        relationships: list[dict],
    ) -> str:
        """构建完整内容"""
        parts = []

        parts.append("## 实体")
        for e in entity_infos:
            parts.append(f"- **{e['name']}** ({e['type']}): {e.get('description', '')}")

        parts.append("\n## 关系")
        for r in relationships:
            parts.append(f"- {r['source']} → {r['target']} ({r['type']})")

        return "\n".join(parts)

    async def search(
        self,
        query: str,
        community_level: int | None = None,
        response_type: str = "多段落",
    ) -> dict[str, Any]:
        """
        执行Global Search

        Args:
            query: 用户问题
            community_level: 社区层级（None表示所有层级）
            response_type: 响应类型

        Returns:
            搜索结果
        """
        # 过滤社区报告
        reports = self.community_reports
        if community_level is not None:
            reports = [r for r in reports if r.level == community_level]

        if not reports:
            return {
                "query": query,
                "answer": "暂无相关社区知识可供参考。",
                "sources": [],
            }

        # Map阶段：对每个社区生成中间回答
        map_results = await self._map_phase(query, reports, response_type)

        # Reduce阶段：整合中间回答
        final_answer = await self._reduce_phase(query, map_results, response_type)

        return {
            "query": query,
            "answer": final_answer,
            "sources": [r.to_dict() for r in reports[:5]],
            "community_count": len(reports),
        }

    async def _map_phase(
        self,
        query: str,
        reports: list[CommunityReport],
        response_type: str,
    ) -> list[str]:
        """Map阶段：处理每个社区"""
        map_results = []

        # 按批次处理（避免过多并发）
        batch_size = min(self.config.concurrency, len(reports))
        for i in range(0, len(reports), batch_size):
            batch = reports[i : i + batch_size]

            context_data = "\n\n".join(
                [f"### {r.title}\n{r.summary}\n\n{r.full_content}" for r in batch]
            )

            prompt = GLOBAL_SEARCH_MAP_PROMPT.format(
                context_data=context_data[: self.config.map_max_tokens * 4],
                response_type=response_type,
            )

            try:
                result = await self.llm_client.chat_completion(
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": query},
                    ]
                )
                map_results.append(result)
            except Exception as e:
                logger.error("Map phase failed", error=str(e))

        return map_results

    async def _reduce_phase(
        self,
        query: str,
        map_results: list[str],
        response_type: str,
    ) -> str:
        """Reduce阶段：整合结果"""
        if not map_results:
            return "无法生成回答。"

        if len(map_results) == 1:
            return map_results[0]

        report_data = "\n\n---\n\n".join(
            [f"报告 {i + 1}:\n{result}" for i, result in enumerate(map_results)]
        )

        prompt = GLOBAL_SEARCH_REDUCE_PROMPT.format(
            query=query,
            report_data=report_data[: self.config.reduce_max_tokens * 4],
            response_type=response_type,
        )

        try:
            final_answer = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )
            return final_answer
        except Exception as e:
            logger.error("Reduce phase failed", error=str(e))
            return map_results[0]


# Local Search Prompt
LOCAL_SEARCH_PROMPT = """---角色---
你是一个专业的文博领域知识助手。

---目标---
根据提供的知识图谱上下文和文本数据回答用户问题。

---上下文数据---

## 实体
{entities}

## 关系
{relationships}

## 相关文本
{text_units}

---用户问题---
{query}

---回答要求---
1. 基于提供的上下文回答
2. 如果上下文不足以回答，说明需要更多信息
3. 不要编造上下文中没有的信息
4. 使用markdown格式

请回答："""


class GraphRAGLocalSearcher:
    """
    GraphRAG Local Search实现

    基于实体及其邻域的精确搜索策略
    适用于具体问题，如"四羊方尊是什么材质的"
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        graph_store: GraphStore,
        llm_client: Any,
        config: GraphRAGConfig | None = None,
    ):
        """
        初始化Local Searcher

        Args:
            vector_retriever: 向量检索器
            graph_store: 图存储
            llm_client: LLM客户端
            config: GraphRAG配置
        """
        self.vector_retriever = vector_retriever
        self.graph_store = graph_store
        self.llm_client = llm_client
        self.config = config or GraphRAGConfig.from_settings()

    async def search(
        self,
        query: str,
        top_k: int = 10,
        include_neighbors: bool = True,
        neighbor_depth: int = 1,
    ) -> dict[str, Any]:
        """
        执行Local Search

        Args:
            query: 用户问题
            top_k: 返回数量
            include_neighbors: 是否包含邻居实体
            neighbor_depth: 邻居搜索深度

        Returns:
            搜索结果
        """
        # 1. 向量检索获取相关文本块
        vector_results = await self.vector_retriever.retrieve(
            query,
            top_k=self.config.local_search_text_unit_count,
        )

        # 2. 从检索结果中提取实体名称
        entity_names = set()
        for result in vector_results:
            if artifact_name := result.metadata.get("artifact_name"):
                entity_names.add(artifact_name)
            if entity_name := result.metadata.get("entity_name"):
                entity_names.add(entity_name)

        # 3. 搜索图中的相关实体
        all_entities: list[Entity] = []
        all_relationships: list[dict] = []

        for entity_name in list(entity_names)[:5]:
            # 搜索匹配的实体
            entities = await self.graph_store.search_entities(entity_name, limit=3)
            all_entities.extend(entities)

            # 获取邻居
            if include_neighbors and entities:
                for entity in entities[:1]:  # 只取第一个匹配
                    neighbors = await self.graph_store.get_neighbors(
                        entity.id,
                        depth=neighbor_depth,
                    )
                    for neighbor in neighbors:
                        all_relationships.append(
                            {
                                "source": entity.name,
                                "target": neighbor["entity"]["name"],
                                "type": ", ".join(neighbor.get("relation_types", [])),
                            }
                        )

        # 4. 构建上下文
        context = self._build_context(
            all_entities,
            all_relationships,
            vector_results,
        )

        # 5. 生成回答
        answer = await self._generate_answer(query, context)

        return {
            "query": query,
            "answer": answer,
            "entities": [self._entity_to_dict(e) for e in all_entities[:top_k]],
            "chunks": [r.to_dict() for r in vector_results],
            "relationships": all_relationships[:20],
        }

    def _build_context(
        self,
        entities: list[Entity],
        relationships: list[dict],
        text_results: list[RetrievalResult],
    ) -> dict[str, str]:
        """构建上下文"""
        # 实体信息
        entities_text = (
            "\n".join(
                [
                    f"- **{e.name}** ({e.type.value if hasattr(e.type, 'value') else e.type}): "
                    f"{e.description[:200] if e.description else '无描述'}"
                    for e in entities[:15]
                ]
            )
            or "无相关实体"
        )

        # 关系信息
        relationships_text = (
            "\n".join(
                [f"- {r['source']} --[{r['type']}]--> {r['target']}" for r in relationships[:15]]
            )
            or "无相关关系"
        )

        # 文本块
        text_units = (
            "\n\n".join(
                [
                    f"[来源: {r.metadata.get('source', '未知')}]\n{r.content[:500]}"
                    for r in text_results[:10]
                ]
            )
            or "无相关文本"
        )

        return {
            "entities": entities_text,
            "relationships": relationships_text,
            "text_units": text_units,
        }

    async def _generate_answer(
        self,
        query: str,
        context: dict[str, str],
    ) -> str:
        """生成回答"""
        prompt = LOCAL_SEARCH_PROMPT.format(
            entities=context["entities"],
            relationships=context["relationships"],
            text_units=context["text_units"],
            query=query,
        )

        try:
            answer = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )
            return answer
        except Exception as e:
            logger.error("Local search answer generation failed", error=str(e))
            return "抱歉，无法生成回答。"

    def _entity_to_dict(self, entity: Entity) -> dict[str, Any]:
        """实体转字典"""
        return {
            "id": entity.id,
            "name": entity.name,
            "type": entity.type.value if hasattr(entity.type, "value") else str(entity.type),
            "description": entity.description,
        }
