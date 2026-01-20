"""
向量检索器

基于向量相似度的语义检索
"""

from dataclasses import dataclass, field
from typing import Any

from kg_search.indexing.vector_store import VectorStore
from kg_search.utils import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """检索结果"""

    id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    source_type: str = "vector"  # vector, graph, hybrid

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "source_type": self.source_type,
        }


class VectorRetriever:
    """向量检索器"""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: Any,
    ):
        """
        初始化向量检索器

        Args:
            vector_store: 向量存储
            embedding_service: 嵌入服务
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        """
        向量检索

        Args:
            query: 查询文本
            top_k: 返回数量
            filters: 过滤条件
            min_score: 最小相似度分数

        Returns:
            检索结果列表
        """
        # 生成查询向量
        query_embedding = await self.embedding_service.embed_text(query)

        # 向量搜索
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )

        # 转换结果
        retrieval_results = []
        for item in results:
            if item["score"] >= min_score:
                retrieval_results.append(
                    RetrievalResult(
                        id=item["id"],
                        content=item["content"],
                        score=item["score"],
                        metadata=item["metadata"],
                        source_type="vector",
                    )
                )

        logger.info(
            "Vector retrieval completed",
            query_length=len(query),
            results_count=len(retrieval_results),
        )

        return retrieval_results

    async def retrieve_by_artifact(
        self,
        query: str,
        artifact_name: str | None = None,
        dynasty: str | None = None,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """
        按文物属性过滤的检索

        Args:
            query: 查询文本
            artifact_name: 文物名称过滤
            dynasty: 朝代过滤
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        filters = {}
        if artifact_name:
            filters["artifact_name"] = artifact_name
        if dynasty:
            filters["dynasty"] = dynasty

        return await self.retrieve(query, top_k=top_k, filters=filters if filters else None)

    async def retrieve_similar(
        self,
        chunk_id: str,
        top_k: int = 10,
        exclude_self: bool = True,
    ) -> list[RetrievalResult]:
        """
        查找相似内容

        Args:
            chunk_id: 参考块ID
            top_k: 返回数量
            exclude_self: 是否排除自身

        Returns:
            检索结果列表
        """
        # 获取参考块的内容
        chunks = await self.vector_store.get_by_ids([chunk_id])
        if not chunks:
            return []

        reference_content = chunks[0]["content"]

        # 搜索相似内容
        results = await self.retrieve(
            reference_content,
            top_k=top_k + 1 if exclude_self else top_k,
        )

        # 排除自身
        if exclude_self:
            results = [r for r in results if r.id != chunk_id][:top_k]

        return results
