"""
GraphRAG Local Search

针对具体实体的精确检索策略
"""

from typing import Any

from kg_search.config import get_settings
from kg_search.retrieval.graph_retriever import GraphRetriever
from kg_search.retrieval.vector_retriever import RetrievalResult, VectorRetriever
from kg_search.utils import get_logger

logger = get_logger(__name__)


class LocalSearch:
    """
    GraphRAG Local Search

    适用于具体问题，如：
    - "四羊方尊是什么朝代的？"
    - "这件文物的材质是什么？"
    - "和这件文物同时期的还有什么？"

    支持两种模式：
    1. 原生模式：使用自研的检索和生成
    2. GraphRAG模式：使用GraphRAG适配器的实现
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        graph_retriever: GraphRetriever,
        llm_client: Any,
        use_graphrag: bool | None = None,
    ):
        """
        初始化Local Search

        Args:
            vector_retriever: 向量检索器
            graph_retriever: 图检索器
            llm_client: LLM客户端
            use_graphrag: 是否使用GraphRAG模式（None表示从配置读取）
        """
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.llm_client = llm_client

        settings = get_settings()
        self._use_graphrag = (
            use_graphrag if use_graphrag is not None else settings.use_graphrag_searcher
        )

        if self._use_graphrag:
            from kg_search.graphrag import GraphRAGLocalSearcher

            self._graphrag_searcher = GraphRAGLocalSearcher(
                vector_retriever=vector_retriever,
                graph_store=graph_retriever.graph_store,
                llm_client=llm_client,
            )
            logger.info("LocalSearch using GraphRAG mode")
        else:
            self._graphrag_searcher = None
            logger.info("LocalSearch using native mode")

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
            query: 查询文本
            top_k: 返回数量
            include_neighbors: 是否包含邻居实体
            neighbor_depth: 邻居搜索深度

        Returns:
            搜索结果，包含实体、关系和文本块
        """
        # 如果使用GraphRAG模式
        if self._use_graphrag and self._graphrag_searcher:
            return await self._graphrag_searcher.search(
                query,
                top_k=top_k,
                include_neighbors=include_neighbors,
                neighbor_depth=neighbor_depth,
            )

        # 原生模式
        results = {
            "query": query,
            "entities": [],
            "chunks": [],
            "context": "",
        }

        # 1. 向量检索获取相关文本块
        vector_results = await self.vector_retriever.retrieve(query, top_k=top_k)
        results["chunks"] = [r.to_dict() for r in vector_results]

        # 2. 从检索结果中提取实体名称
        entity_names = set()
        for result in vector_results:
            if artifact_name := result.metadata.get("artifact_name"):
                entity_names.add(artifact_name)

        # 3. 图检索获取实体及其关系
        all_entities = []
        for entity_name in list(entity_names)[:5]:
            entity_results = await self.graph_retriever.retrieve_by_entity(
                entity_name,
                depth=neighbor_depth if include_neighbors else 0,
                limit=10,
            )
            all_entities.extend(entity_results)

        # 去重
        seen_ids = set()
        unique_entities = []
        for entity in all_entities:
            if entity.id not in seen_ids:
                seen_ids.add(entity.id)
                unique_entities.append(entity)

        results["entities"] = [e.to_dict() for e in unique_entities[:top_k]]

        # 4. 构建上下文
        results["context"] = self._build_context(vector_results, unique_entities)

        logger.info(
            "Local search completed",
            query=query,
            chunks_count=len(results["chunks"]),
            entities_count=len(results["entities"]),
        )

        return results

    async def search_with_answer(
        self,
        query: str,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """
        执行Local Search并生成答案

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            包含答案的搜索结果
        """
        # 执行搜索
        search_results = await self.search(query, top_k=top_k)

        # 生成答案
        answer = await self._generate_answer(query, search_results["context"])
        search_results["answer"] = answer

        return search_results

    def _build_context(
        self,
        chunks: list[RetrievalResult],
        entities: list[RetrievalResult],
    ) -> str:
        """
        构建上下文文本

        Args:
            chunks: 文本块列表
            entities: 实体列表

        Returns:
            格式化的上下文
        """
        context_parts = []

        # 添加实体信息
        if entities:
            context_parts.append("## 相关实体信息\n")
            for entity in entities[:10]:
                context_parts.append(f"- {entity.content}")
            context_parts.append("")

        # 添加文本块
        if chunks:
            context_parts.append("## 相关文档内容\n")
            for i, chunk in enumerate(chunks[:5], 1):
                context_parts.append(f"### 文档片段 {i}")
                context_parts.append(chunk.content)
                context_parts.append("")

        return "\n".join(context_parts)

    async def _generate_answer(self, query: str, context: str) -> str:
        """
        生成答案

        Args:
            query: 用户问题
            context: 检索到的上下文

        Returns:
            生成的答案
        """
        prompt = f"""基于以下上下文信息回答用户的问题。如果上下文中没有相关信息，请说明无法回答。

## 上下文
{context}

## 用户问题
{query}

## 回答要求
1. 基于上下文中的信息回答
2. 如果涉及文物，请提供详细的属性信息
3. 如果有关联的文物或实体，可以适当提及
4. 保持回答准确、简洁

请回答："""

        try:
            answer = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )
            return answer
        except Exception as e:
            logger.error("Answer generation failed", error=str(e))
            return "抱歉，生成答案时出现错误。"
