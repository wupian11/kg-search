"""
混合检索器

结合向量检索和图检索的混合检索策略
"""

from typing import Any

from kg_search.config import get_settings
from kg_search.utils import get_logger

from .graph_retriever import GraphRetriever
from .vector_retriever import RetrievalResult, VectorRetriever

logger = get_logger(__name__)


class HybridRetriever:
    """混合检索器"""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        graph_retriever: GraphRetriever,
        alpha: float | None = None,
    ):
        """
        初始化混合检索器

        Args:
            vector_retriever: 向量检索器
            graph_retriever: 图检索器
            alpha: 向量检索权重 (0-1)，图检索权重为 1-alpha
        """
        settings = get_settings()

        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.alpha = alpha if alpha is not None else settings.hybrid_search_alpha

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        vector_top_k: int | None = None,
        graph_top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        entity_types: list[str] | None = None,
        fusion_method: str = "rrf",  # rrf, weighted, interleave
    ) -> list[RetrievalResult]:
        """
        混合检索

        Args:
            query: 查询文本
            top_k: 最终返回数量
            vector_top_k: 向量检索数量
            graph_top_k: 图检索数量
            filters: 向量检索过滤条件
            entity_types: 图检索实体类型过滤
            fusion_method: 融合方法 (rrf/weighted/interleave)

        Returns:
            融合后的检索结果
        """
        vector_top_k = vector_top_k or top_k * 2
        graph_top_k = graph_top_k or top_k * 2

        # 并行执行向量检索和图检索
        vector_results = await self.vector_retriever.retrieve(
            query,
            top_k=vector_top_k,
            filters=filters,
        )

        graph_results = await self.graph_retriever.retrieve_by_query(
            query,
            entity_types=entity_types,
            limit=graph_top_k,
        )

        # 融合结果
        if fusion_method == "rrf":
            results = self._rrf_fusion(vector_results, graph_results, top_k)
        elif fusion_method == "weighted":
            results = self._weighted_fusion(vector_results, graph_results, top_k)
        else:  # interleave
            results = self._interleave_fusion(vector_results, graph_results, top_k)

        logger.info(
            "Hybrid retrieval completed",
            query_length=len(query),
            vector_count=len(vector_results),
            graph_count=len(graph_results),
            final_count=len(results),
            fusion_method=fusion_method,
        )

        return results

    async def retrieve_with_expansion(
        self,
        query: str,
        top_k: int = 10,
        expansion_depth: int = 1,
    ) -> list[RetrievalResult]:
        """
        带图扩展的检索

        先进行向量检索，然后基于检索到的实体进行图扩展

        Args:
            query: 查询文本
            top_k: 返回数量
            expansion_depth: 图扩展深度

        Returns:
            扩展后的检索结果
        """
        # 1. 向量检索
        vector_results = await self.vector_retriever.retrieve(query, top_k=top_k)

        # 2. 从向量结果中提取实体名称
        entity_names = set()
        for result in vector_results:
            if artifact_name := result.metadata.get("artifact_name"):
                entity_names.add(artifact_name)

        # 3. 图扩展
        graph_results = []
        for entity_name in list(entity_names)[:5]:  # 限制扩展数量
            expanded = await self.graph_retriever.retrieve_by_entity(
                entity_name,
                depth=expansion_depth,
                limit=5,
            )
            graph_results.extend(expanded)

        # 4. 融合结果
        results = self._rrf_fusion(vector_results, graph_results, top_k)

        return results

    def _rrf_fusion(
        self,
        vector_results: list[RetrievalResult],
        graph_results: list[RetrievalResult],
        top_k: int,
        k: int = 60,
    ) -> list[RetrievalResult]:
        """
        Reciprocal Rank Fusion (RRF) 融合

        RRF Score = Σ 1 / (k + rank)

        Args:
            vector_results: 向量检索结果
            graph_results: 图检索结果
            top_k: 返回数量
            k: RRF参数

        Returns:
            融合后的结果
        """
        scores: dict[str, float] = {}
        result_map: dict[str, RetrievalResult] = {}

        # 计算向量检索的RRF分数
        for rank, result in enumerate(vector_results):
            rrf_score = 1.0 / (k + rank + 1)
            scores[result.id] = scores.get(result.id, 0) + rrf_score * self.alpha
            result_map[result.id] = result

        # 计算图检索的RRF分数
        for rank, result in enumerate(graph_results):
            rrf_score = 1.0 / (k + rank + 1)
            scores[result.id] = scores.get(result.id, 0) + rrf_score * (1 - self.alpha)
            if result.id not in result_map:
                result_map[result.id] = result

        # 按融合分数排序
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        results = []
        for id_ in sorted_ids[:top_k]:
            result = result_map[id_]
            result.score = scores[id_]
            result.source_type = "hybrid"
            results.append(result)

        return results

    def _weighted_fusion(
        self,
        vector_results: list[RetrievalResult],
        graph_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        加权分数融合

        Args:
            vector_results: 向量检索结果
            graph_results: 图检索结果
            top_k: 返回数量

        Returns:
            融合后的结果
        """
        scores: dict[str, float] = {}
        result_map: dict[str, RetrievalResult] = {}

        # 向量检索分数（已归一化）
        for result in vector_results:
            scores[result.id] = result.score * self.alpha
            result_map[result.id] = result

        # 图检索分数
        for result in graph_results:
            current_score = scores.get(result.id, 0)
            scores[result.id] = current_score + result.score * (1 - self.alpha)
            if result.id not in result_map:
                result_map[result.id] = result

        # 排序
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        results = []
        for id_ in sorted_ids[:top_k]:
            result = result_map[id_]
            result.score = scores[id_]
            result.source_type = "hybrid"
            results.append(result)

        return results

    def _interleave_fusion(
        self,
        vector_results: list[RetrievalResult],
        graph_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        交错融合

        按比例交错排列两种结果

        Args:
            vector_results: 向量检索结果
            graph_results: 图检索结果
            top_k: 返回数量

        Returns:
            融合后的结果
        """
        results = []
        seen_ids = set()

        v_idx = 0
        g_idx = 0

        while len(results) < top_k:
            # 根据alpha决定下一个取哪个来源
            take_vector = (len(results) % 2 == 0) if self.alpha >= 0.5 else (len(results) % 2 == 1)

            if take_vector and v_idx < len(vector_results):
                result = vector_results[v_idx]
                v_idx += 1
                if result.id not in seen_ids:
                    seen_ids.add(result.id)
                    result.source_type = "hybrid"
                    results.append(result)
            elif g_idx < len(graph_results):
                result = graph_results[g_idx]
                g_idx += 1
                if result.id not in seen_ids:
                    seen_ids.add(result.id)
                    result.source_type = "hybrid"
                    results.append(result)
            elif v_idx < len(vector_results):
                result = vector_results[v_idx]
                v_idx += 1
                if result.id not in seen_ids:
                    seen_ids.add(result.id)
                    result.source_type = "hybrid"
                    results.append(result)
            else:
                break

        return results
