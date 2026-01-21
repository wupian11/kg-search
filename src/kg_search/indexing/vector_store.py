"""
向量存储

使用ChromaDB存储文本向量 - 支持台湾文物数据的多维度过滤
"""

from abc import ABC, abstractmethod
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from kg_search.config import get_settings
from kg_search.ingestion.chunkers.base import Chunk
from kg_search.utils import get_logger

logger = get_logger(__name__)


# ============================================================
# 可过滤字段定义
# ============================================================

FILTERABLE_STRING_FIELDS = [
    "artifact_id",  # 文物编号
    "artifact_name",  # 文物名称
    "assets_classify_code",  # 资产分级代码
    "assets_classify_name",  # 资产分级名称
    "assets_type_code",  # 资产类型代码
    "assets_type_name",  # 资产类型名称
    "dynasty",  # 朝代
    "city",  # 城市
    "district",  # 区
    "material",  # 材质
    "creator",  # 创作者
    "keeper",  # 保管单位
    "save_space",  # 存放空间
    "gov_institution",  # 主管机关
    "past_history_source",  # 来源方式
    "reserve_status",  # 保存状态
    "view_type",  # 视图类型
]

FILTERABLE_INT_FIELDS = [
    "year_ce",  # 西元年份
    "year_ce_end",  # 西元年份结束
    "chunk_index",  # 块索引
]


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
    """ChromaDB向量存储 - 支持台湾文物多维度过滤"""

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

        # 使用新的 to_filter_metadata 方法获取完整的可过滤元数据
        metadatas = [chunk.to_filter_metadata() for chunk in chunks]

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
            filters: 过滤条件，支持以下字段：
                - artifact_id, artifact_name: 文物标识
                - assets_classify_code/name, assets_type_code/name: 分类
                - dynasty, year_ce, year_ce_end: 时间
                - city, district: 地理
                - material, creator: 物理属性
                - keeper, save_space, gov_institution: 收藏管理
                - view_type: 视图类型

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

    async def search_by_year_range(
        self,
        query_embedding: list[float],
        year_start: int | None = None,
        year_end: int | None = None,
        top_k: int = 10,
        additional_filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        按年代范围搜索

        Args:
            query_embedding: 查询向量
            year_start: 起始年份（西元）
            year_end: 结束年份（西元）
            top_k: 返回数量
            additional_filters: 额外过滤条件

        Returns:
            搜索结果列表
        """
        filters = additional_filters.copy() if additional_filters else {}

        if year_start is not None:
            filters["year_ce_gte"] = year_start
        if year_end is not None:
            filters["year_ce_lte"] = year_end

        return await self.search(query_embedding, top_k, filters)

    async def search_by_location(
        self,
        query_embedding: list[float],
        city: str | None = None,
        district: str | None = None,
        top_k: int = 10,
        additional_filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        按地理位置搜索

        Args:
            query_embedding: 查询向量
            city: 城市名
            district: 区名
            top_k: 返回数量
            additional_filters: 额外过滤条件

        Returns:
            搜索结果列表
        """
        filters = additional_filters.copy() if additional_filters else {}

        if city:
            filters["city"] = city
        if district:
            filters["district"] = district

        return await self.search(query_embedding, top_k, filters)

    async def search_by_view_type(
        self,
        query_embedding: list[float],
        view_type: str,
        top_k: int = 10,
        additional_filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        按视图类型搜索（多视图embedding策略）

        Args:
            query_embedding: 查询向量
            view_type: 视图类型（full, basic, description, historical, physical, location）
            top_k: 返回数量
            additional_filters: 额外过滤条件

        Returns:
            搜索结果列表
        """
        filters = additional_filters.copy() if additional_filters else {}
        filters["view_type"] = view_type

        return await self.search(query_embedding, top_k, filters)

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
        """
        构建ChromaDB的where子句

        支持的过滤操作：
        - 精确匹配: {"field": "value"}
        - 列表匹配: {"field": ["v1", "v2"]}
        - 范围查询: {"field_gte": 1900, "field_lte": 2000}
        """
        conditions = []

        for key, value in filters.items():
            # 处理范围查询
            if key.endswith("_gte"):
                field = key[:-4]  # 移除 _gte 后缀
                conditions.append({field: {"$gte": value}})
            elif key.endswith("_lte"):
                field = key[:-4]  # 移除 _lte 后缀
                conditions.append({field: {"$lte": value}})
            elif key.endswith("_gt"):
                field = key[:-3]
                conditions.append({field: {"$gt": value}})
            elif key.endswith("_lt"):
                field = key[:-3]
                conditions.append({field: {"$lt": value}})
            elif key.endswith("_ne"):
                field = key[:-3]
                conditions.append({field: {"$ne": value}})
            # 处理列表匹配
            elif isinstance(value, list):
                conditions.append({key: {"$in": value}})
            # 处理精确匹配
            else:
                conditions.append({key: {"$eq": value}})

        if len(conditions) == 0:
            return {}
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def get_collection_stats(self) -> dict[str, Any]:
        """获取集合统计信息"""
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
        }

    async def get_distinct_values(self, field: str, limit: int = 100) -> list[str]:
        """
        获取某字段的去重值列表（用于构建过滤器UI）

        Args:
            field: 字段名
            limit: 返回数量限制

        Returns:
            去重后的值列表
        """
        # ChromaDB 不直接支持 distinct，需要获取所有记录后处理
        # 这里提供一个基础实现，生产环境可能需要优化
        results = self.collection.get(
            limit=limit * 10,  # 获取更多记录以提高覆盖率
            include=["metadatas"],
        )

        values = set()
        if results["metadatas"]:
            for metadata in results["metadatas"]:
                if field in metadata and metadata[field]:
                    values.add(metadata[field])
                    if len(values) >= limit:
                        break

        return sorted(list(values))
