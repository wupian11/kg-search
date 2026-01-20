"""
向量存储

使用ChromaDB存储文本向量
"""

from abc import ABC, abstractmethod
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from kg_search.config import get_settings
from kg_search.ingestion.chunkers.base import Chunk
from kg_search.utils import get_logger

logger = get_logger(__name__)


class VectorStore(ABC):
    """向量存储抽象基类"""

    @abstractmethod
    async def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """添加文本块及其向量"""
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """向量相似度搜索"""
        pass

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """删除向量"""
        pass

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """根据ID获取"""
        pass


class ChromaVectorStore(VectorStore):
    """ChromaDB向量存储"""

    def __init__(
        self,
        collection_name: str | None = None,
        persist_directory: str | None = None,
    ):
        """
        初始化ChromaDB存储

        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录
        """
        settings = get_settings()

        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_directory = persist_directory or settings.chroma_persist_directory

        # 初始化ChromaDB客户端
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
        )

        logger.info(
            "ChromaDB initialized",
            collection=self.collection_name,
            persist_dir=self.persist_directory,
        )

    async def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """
        添加文本块及其向量

        Args:
            chunks: 文本块列表
            embeddings: 对应的向量列表
        """
        if not chunks:
            return

        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "artifact_id": chunk.artifact_id or "",
                "artifact_name": chunk.artifact_name or "",
                "dynasty": chunk.dynasty or "",
            }
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info("Added chunks to ChromaDB", count=len(chunks))

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        向量相似度搜索

        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            搜索结果列表
        """
        where = None
        if filters:
            where = self._build_where_clause(filters)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # 转换结果格式
        items = []
        if results["ids"] and results["ids"][0]:
            for i, id_ in enumerate(results["ids"][0]):
                item = {
                    "id": id_,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": 1 - results["distances"][0][i] if results["distances"] else 0,
                }
                items.append(item)

        return items

    async def delete(self, ids: list[str]) -> None:
        """删除向量"""
        if ids:
            self.collection.delete(ids=ids)
            logger.info("Deleted from ChromaDB", count=len(ids))

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """根据ID获取"""
        if not ids:
            return []

        results = self.collection.get(
            ids=ids,
            include=["documents", "metadatas"],
        )

        items = []
        if results["ids"]:
            for i, id_ in enumerate(results["ids"]):
                item = {
                    "id": id_,
                    "content": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                }
                items.append(item)

        return items

    def _build_where_clause(self, filters: dict[str, Any]) -> dict[str, Any]:
        """构建ChromaDB的where子句"""
        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append({key: {"$in": value}})
            else:
                conditions.append({key: {"$eq": value}})

        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def get_collection_stats(self) -> dict[str, Any]:
        """获取集合统计信息"""
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
        }
