"""
文本分块器基类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kg_search.ingestion.loaders.base import Document


class ViewType(str, Enum):
    """多视图类型枚举"""

    FULL = "full"  # 完整视图（包含所有字段）
    BASIC = "basic"  # 基础信息视图
    DESCRIPTION = "description"  # 详细描述视图
    HISTORICAL = "historical"  # 历史背景视图
    PHYSICAL = "physical"  # 物理属性视图
    LOCATION = "location"  # 地理位置视图


@dataclass
class Chunk:
    """文本块数据结构 - 支持台湾文物数据的多维度元数据"""

    id: str
    content: str
    document_id: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    # ============================================================
    # 基础标识字段
    # ============================================================
    artifact_id: str | None = None  # caseId - 文物编号
    artifact_name: str | None = None  # caseName - 文物名称
    artifact_name_normalized: str | None = None  # 标准化名称（繁体）

    # ============================================================
    # 分类信息
    # ============================================================
    assets_classify_code: str | None = None  # assetsClassifyCode - 资产分级代码
    assets_classify_name: str | None = None  # assetsClassifyName - 资产分级名称（如一般古物）
    assets_type_code: str | None = None  # assetsTypes.subCode - 资产类型代码（如G3.1）
    assets_type_name: str | None = None  # assetsTypes.subName - 资产类型名称（如圖書、報刊）

    # ============================================================
    # 时间信息
    # ============================================================
    dynasty: str | None = None  # 朝代/时期（如日治時期、民國）
    age_description: str | None = None  # descAge - 原始年代描述
    year_ce: int | None = None  # 标准化西元年份
    year_ce_end: int | None = None  # 西元年份结束（用于时间范围）

    # ============================================================
    # 地理信息
    # ============================================================
    city: str | None = None  # addresses.cityName - 城市
    district: str | None = None  # addresses.distName - 区
    address: str | None = None  # addresses.address - 详细地址
    full_address: str | None = None  # 完整地址拼接

    # ============================================================
    # 物理属性
    # ============================================================
    material: str | None = None  # descMaterial - 材质
    size: str | None = None  # descSize - 尺寸
    creator: str | None = None  # descCreator - 创作者

    # ============================================================
    # 收藏保管信息
    # ============================================================
    keeper: str | None = None  # keepDepts.name - 保管单位
    save_space: str | None = None  # keepPlaces.saveSpace - 存放空间类型
    save_space_identity: str | None = None  # keepPlaces.saveSpaceIdentity - 空间文资身分

    # ============================================================
    # 管理信息
    # ============================================================
    gov_institution: str | None = None  # govInstitution - 主管机关
    gov_dept_name: str | None = None  # govDeptName - 业务科室
    reservation_code: str | None = None  # reservationCode - 公告字号

    # ============================================================
    # 来源与状态
    # ============================================================
    past_history_source: str | None = None  # pastHistorySource - 来源方式
    reserve_status: str | None = None  # reserveStatus - 保存状态

    # ============================================================
    # 多视图支持
    # ============================================================
    view_type: ViewType | str = ViewType.FULL  # 视图类型

    # ============================================================
    # 溯源信息
    # ============================================================
    source_fields: list[str] = field(default_factory=list)  # 构成此chunk的源字段列表

    # 块的位置信息
    start_char: int = 0
    end_char: int = 0

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
            # 基础标识
            "artifact_id": self.artifact_id,
            "artifact_name": self.artifact_name,
            "artifact_name_normalized": self.artifact_name_normalized,
            # 分类信息
            "assets_classify_code": self.assets_classify_code,
            "assets_classify_name": self.assets_classify_name,
            "assets_type_code": self.assets_type_code,
            "assets_type_name": self.assets_type_name,
            # 时间信息
            "dynasty": self.dynasty,
            "age_description": self.age_description,
            "year_ce": self.year_ce,
            "year_ce_end": self.year_ce_end,
            # 地理信息
            "city": self.city,
            "district": self.district,
            "address": self.address,
            "full_address": self.full_address,
            # 物理属性
            "material": self.material,
            "size": self.size,
            "creator": self.creator,
            # 收藏保管
            "keeper": self.keeper,
            "save_space": self.save_space,
            "save_space_identity": self.save_space_identity,
            # 管理信息
            "gov_institution": self.gov_institution,
            "gov_dept_name": self.gov_dept_name,
            "reservation_code": self.reservation_code,
            # 来源与状态
            "past_history_source": self.past_history_source,
            "reserve_status": self.reserve_status,
            # 视图与溯源
            "view_type": self.view_type.value
            if isinstance(self.view_type, ViewType)
            else self.view_type,
            "source_fields": self.source_fields,
            # 位置信息
            "start_char": self.start_char,
            "end_char": self.end_char,
        }

    def to_filter_metadata(self) -> dict[str, Any]:
        """
        转换为用于向量数据库过滤的元数据
        只包含非空的可过滤字段
        """
        metadata = {}

        # 可过滤的字符串字段
        filterable_str_fields = [
            ("artifact_id", self.artifact_id),
            ("artifact_name", self.artifact_name),
            ("assets_classify_code", self.assets_classify_code),
            ("assets_classify_name", self.assets_classify_name),
            ("assets_type_code", self.assets_type_code),
            ("assets_type_name", self.assets_type_name),
            ("dynasty", self.dynasty),
            ("city", self.city),
            ("district", self.district),
            ("material", self.material),
            ("creator", self.creator),
            ("keeper", self.keeper),
            ("save_space", self.save_space),
            ("gov_institution", self.gov_institution),
            ("past_history_source", self.past_history_source),
            ("reserve_status", self.reserve_status),
            (
                "view_type",
                self.view_type.value if isinstance(self.view_type, ViewType) else self.view_type,
            ),
        ]

        for field_name, value in filterable_str_fields:
            if value:
                metadata[field_name] = value

        # 可过滤的整数字段
        if self.year_ce is not None:
            metadata["year_ce"] = self.year_ce
        if self.year_ce_end is not None:
            metadata["year_ce_end"] = self.year_ce_end

        # 必需字段
        metadata["document_id"] = self.document_id
        metadata["chunk_index"] = self.chunk_index

        return metadata


class TextChunker(ABC):
    """文本分块器抽象基类"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        初始化分块器

        Args:
            chunk_size: 块大小（字符数）
            chunk_overlap: 块重叠（字符数）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """
        将文档分块

        Args:
            document: 输入文档

        Returns:
            块列表
        """
        pass

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """
        批量处理文档

        Args:
            documents: 文档列表

        Returns:
            所有块的列表
        """
        chunks = []
        for doc in documents:
            chunks.extend(self.chunk(doc))
        return chunks
