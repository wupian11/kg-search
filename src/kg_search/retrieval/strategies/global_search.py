"""
GraphRAG Global Search

基于社区摘要的全局检索策略
"""

import json
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from kg_search.config import get_settings
from kg_search.extraction.graph_builder import KnowledgeGraph
from kg_search.utils import get_logger

logger = get_logger(__name__)


@dataclass
class Community:
    """社区数据结构"""

    id: str
    level: int
    title: str
    summary: str
    entity_ids: list[str] = field(default_factory=list)
    rating: float = 0.0
    rating_explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "level": self.level,
            "title": self.title,
            "summary": self.summary,
            "entity_ids": self.entity_ids,
            "rating": self.rating,
            "rating_explanation": self.rating_explanation,
        }


class GlobalSearch:
    """
    GraphRAG Global Search

    适用于宏观问题，如：
    - "中国青铜器的发展历程是怎样的？"
    - "商周时期的主要文物类型有哪些？"
    - "不同朝代的玉器有什么特点？"

    支持两种模式：
    1. 原生模式：使用自研的社区检测和摘要生成
    2. GraphRAG模式：使用GraphRAG适配器的实现
    """

    def __init__(self, llm_client: Any, use_graphrag: bool | None = None):
        """
        初始化Global Search

        Args:
            llm_client: LLM客户端
            use_graphrag: 是否使用GraphRAG模式（None表示从配置读取）
        """
        self.llm_client = llm_client
        self.settings = get_settings()
        self.communities: list[Community] = []

        # 判断是否使用GraphRAG
        self._use_graphrag = (
            use_graphrag if use_graphrag is not None else self.settings.use_graphrag_searcher
        )

        if self._use_graphrag:
            from kg_search.graphrag import GraphRAGGlobalSearcher

            self._graphrag_searcher = GraphRAGGlobalSearcher(llm_client)
            logger.info("GlobalSearch using GraphRAG mode")
        else:
            self._graphrag_searcher = None
            logger.info("GlobalSearch using native mode")

    async def build_communities(
        self,
        knowledge_graph: KnowledgeGraph,
        max_community_size: int | None = None,
    ) -> list[Community]:
        """
        构建社区并生成摘要

        Args:
            knowledge_graph: 知识图谱
            max_community_size: 最大社区大小

        Returns:
            社区列表
        """
        max_size = max_community_size or self.settings.community_max_size

        # 转换为NetworkX图
        graph = knowledge_graph.to_networkx()

        if graph.number_of_nodes() == 0:
            logger.warning("Empty knowledge graph, no communities to build")
            return []

        # 执行社区检测
        communities_data = self._detect_communities(graph)

        # 为每个社区生成摘要
        communities = []
        for i, (community_nodes, level) in enumerate(communities_data):
            if len(community_nodes) > max_size:
                continue  # 跳过过大的社区

            # 获取社区中的实体信息
            entity_infos = []
            for node_id in community_nodes:
                if graph.has_node(node_id):
                    node_data = graph.nodes[node_id]
                    entity_infos.append(
                        {
                            "name": node_data.get("name", node_id),
                            "type": node_data.get("type", ""),
                            "description": node_data.get("description", ""),
                        }
                    )

            # 生成社区摘要
            summary = await self._generate_community_summary(entity_infos)
            title = await self._generate_community_title(entity_infos, summary)

            community = Community(
                id=f"community_{i}",
                level=level,
                title=title,
                summary=summary,
                entity_ids=list(community_nodes),
            )
            communities.append(community)

        self.communities = communities
        logger.info("Communities built", count=len(communities))

        return communities

    def _detect_communities(self, graph: nx.Graph) -> list[tuple[set[str], int]]:
        """
        执行社区检测

        Args:
            graph: NetworkX图

        Returns:
            (社区节点集合, 层级) 列表
        """
        if graph.number_of_nodes() < 2:
            return [(set(graph.nodes()), 0)]

        communities = []

        try:
            # 尝试使用Leiden算法
            from graspologic.partition import hierarchical_leiden

            # 转换为适合graspologic的格式
            node_list = list(graph.nodes())

            # 构建邻接矩阵
            adj_matrix = nx.to_numpy_array(graph, nodelist=node_list)

            # 执行层次Leiden
            partition = hierarchical_leiden(
                adj_matrix, max_cluster_size=self.settings.community_max_size
            )

            # 解析结果
            for level, level_partition in enumerate(partition):
                for community_idx in set(level_partition):
                    community_nodes = {
                        node_list[i] for i, c in enumerate(level_partition) if c == community_idx
                    }
                    if len(community_nodes) > 1:
                        communities.append((community_nodes, level))

        except ImportError:
            logger.warning("graspologic not available, using Louvain algorithm")
            # 回退到Louvain算法
            try:
                from networkx.algorithms.community import louvain_communities

                louvain_result = louvain_communities(graph)
                for i, community_nodes in enumerate(louvain_result):
                    communities.append((community_nodes, 0))

            except Exception as e:
                logger.error("Community detection failed", error=str(e))
                # 最后回退：整个图作为一个社区
                communities.append((set(graph.nodes()), 0))

        return communities

    async def _generate_community_summary(self, entity_infos: list[dict]) -> str:
        """
        生成社区摘要

        Args:
            entity_infos: 实体信息列表

        Returns:
            社区摘要
        """
        if not entity_infos:
            return ""

        # 构建实体描述
        entities_text = "\n".join(
            [
                f"- {e['name']} ({e['type']}): {e.get('description', '无描述')}"
                for e in entity_infos[:20]  # 限制数量
            ]
        )

        prompt = f"""请为以下一组相关的文博实体生成一个综合性摘要，描述它们的共同特征、历史背景和文化意义。

## 实体列表
{entities_text}

## 要求
1. 摘要应该概括这组实体的共同主题
2. 如果存在时间跨度，请说明
3. 如果存在地域特征，请说明
4. 突出文化和艺术价值
5. 字数控制在200-300字

请生成摘要："""

        try:
            summary = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )
            return summary.strip()
        except Exception as e:
            logger.error("Summary generation failed", error=str(e))
            # 生成简单摘要
            names = [e["name"] for e in entity_infos[:5]]
            return f"该社区包含以下实体：{', '.join(names)}等。"

    async def _generate_community_title(
        self,
        entity_infos: list[dict],
        summary: str,
    ) -> str:
        """
        生成社区标题

        Args:
            entity_infos: 实体信息列表
            summary: 社区摘要

        Returns:
            社区标题
        """
        prompt = f"""根据以下社区摘要，生成一个简短的标题（10-20字）。

摘要：{summary}

标题："""

        try:
            title = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )
            return title.strip()
        except Exception:
            # 使用第一个实体的类型作为标题
            if entity_infos:
                types = list(set(e["type"] for e in entity_infos if e["type"]))
                return f"{types[0] if types else '文物'}相关"
            return "未分类社区"

    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        执行Global Search

        Args:
            query: 查询文本
            top_k: 返回社区数量

        Returns:
            搜索结果
        """
        # 如果使用GraphRAG模式
        if self._use_graphrag and self._graphrag_searcher:
            result = await self._graphrag_searcher.search(query)
            # 转换为统一格式
            return {
                "query": query,
                "communities": result.get("sources", []),
                "context": "",
                "answer": result.get("answer", ""),
            }

        # 原生模式
        if not self.communities:
            logger.warning("No communities available for global search")
            return {
                "query": query,
                "communities": [],
                "context": "",
                "answer": "知识库尚未构建社区索引，无法进行全局检索。",
            }

        # 对社区进行相关性评分
        scored_communities = await self._score_communities(query)

        # 取top_k个最相关的社区
        top_communities = sorted(
            scored_communities,
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        # 构建上下文
        context = self._build_context([c for c, _ in top_communities])

        # 生成答案
        answer = await self._generate_answer(query, context)

        return {
            "query": query,
            "communities": [c.to_dict() for c, _ in top_communities],
            "context": context,
            "answer": answer,
        }

    async def _score_communities(
        self,
        query: str,
    ) -> list[tuple[Community, float]]:
        """
        对社区进行相关性评分

        Args:
            query: 查询文本

        Returns:
            (社区, 分数) 列表
        """
        scored = []

        for community in self.communities:
            # 使用LLM评估相关性
            prompt = f"""请评估以下社区摘要与用户问题的相关性，返回0-10的分数。

用户问题：{query}

社区标题：{community.title}
社区摘要：{community.summary}

请只返回一个数字分数（0-10）："""

            try:
                response = await self.llm_client.chat_completion(
                    messages=[{"role": "user", "content": prompt}]
                )
                score = float(response.strip())
                score = max(0, min(10, score))  # 限制范围
            except Exception:
                # 简单的关键词匹配作为后备
                score = sum(
                    1 for word in query if word in community.title or word in community.summary
                )

            scored.append((community, score))

        return scored

    def _build_context(self, communities: list[Community]) -> str:
        """构建上下文"""
        parts = ["## 相关知识社区\n"]

        for i, community in enumerate(communities, 1):
            parts.append(f"### {i}. {community.title}")
            parts.append(community.summary)
            parts.append("")

        return "\n".join(parts)

    async def _generate_answer(self, query: str, context: str) -> str:
        """生成答案"""
        prompt = f"""基于以下知识社区的摘要信息，回答用户的问题。这是一个宏观性的问题，需要综合多个社区的信息。

{context}

## 用户问题
{query}

## 回答要求
1. 综合多个社区的信息进行回答
2. 如果是关于历史发展的问题，按时间线组织回答
3. 如果是关于比较的问题，列出不同方面的异同
4. 保持回答全面但不冗长

请回答："""

        try:
            answer = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )
            return answer
        except Exception as e:
            logger.error("Answer generation failed", error=str(e))
            return "抱歉，生成答案时出现错误。"

    def save_communities(self, file_path: str) -> None:
        """保存社区数据"""
        data = [c.to_dict() for c in self.communities]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Communities saved", path=file_path)

    def load_communities(self, file_path: str) -> None:
        """加载社区数据"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.communities = []
        for item in data:
            self.communities.append(
                Community(
                    id=item["id"],
                    level=item["level"],
                    title=item["title"],
                    summary=item["summary"],
                    entity_ids=item.get("entity_ids", []),
                    rating=item.get("rating", 0.0),
                    rating_explanation=item.get("rating_explanation", ""),
                )
            )

        logger.info("Communities loaded", count=len(self.communities))
