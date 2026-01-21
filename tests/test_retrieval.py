"""
检索服务层测试
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from kg_search.retrieval import VectorRetriever, GraphRetriever, HybridRetriever
from kg_search.utils.models import SearchResult


class TestVectorRetriever:
    """向量检索测试"""

    @pytest.mark.asyncio
    async def test_search(self, sample_chunks):
        """测试向量搜索"""
        # Mock vector store
        mock_store = MagicMock()
        mock_store.search = AsyncMock(
            return_value=[
                (sample_chunks[0], 0.95),
                (sample_chunks[1], 0.85),
            ]
        )

        # Mock embedding service
        mock_embedding = MagicMock()
        mock_embedding.embed_query = AsyncMock(return_value=[0.1] * 1536)

        retriever = VectorRetriever(mock_store, mock_embedding)

        results = await retriever.search("青铜礼器", top_k=10)

        assert len(results) == 2
        assert results[0].score >= results[1].score


class TestHybridRetriever:
    """混合检索测试"""

    @pytest.mark.asyncio
    async def test_search_rrf_fusion(self, sample_chunks):
        """测试RRF融合搜索"""
        # Mock retrievers
        mock_vector = MagicMock()
        mock_vector.search = AsyncMock(
            return_value=[
                SearchResult(
                    id=sample_chunks[0].id,
                    content=sample_chunks[0].content,
                    score=0.9,
                    metadata={},
                    source="vector",
                )
            ]
        )

        mock_graph = MagicMock()
        mock_graph.search = AsyncMock(
            return_value=[
                SearchResult(
                    id=sample_chunks[1].id,
                    content=sample_chunks[1].content,
                    score=0.8,
                    metadata={},
                    source="graph",
                )
            ]
        )

        retriever = HybridRetriever(mock_vector, mock_graph)

        results = await retriever.search(query="青铜礼器", top_k=10, fusion_method="rrf")

        assert len(results) == 2
