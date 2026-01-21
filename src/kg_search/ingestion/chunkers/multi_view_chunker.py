"""
多视图Embedding策略

为同一文物创建多个视图的embedding，支持多维度检索
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from kg_search.ingestion.chunkers.base import Chunk, ViewType
from kg_search.utils import generate_id, get_logger

logger = get_logger(__name__)


class MultiViewChunker:
    """
    多视图分块器

    为每个文物创建多个不同视角的文本块:
    - FULL: 完整视图，包含所有字段
    - BASIC: 基础信息视图（名称、类型、年代、材质）
    - DESCRIPTION: 详细描述视图（保存状态描述、登录原因）
    - HISTORICAL: 历史背景视图（年代、来源、历史事件）
    - PHYSICAL: 物理属性视图（材质、尺寸、工艺）
    - LOCATION: 地理位置视图（地址、保管单位、存放空间）
    """

    def __init__(
        self,
        views: list[ViewType] | None = None,
        include_full_view: bool = True,
    ):
        """
        初始化多视图分块器

        Args:
            views: 要生成的视图类型列表，默认生成所有视图
            include_full_view: 是否包含完整视图
        """
        if views is None:
            views = list(ViewType)
        self.views = views
        self.include_full_view = include_full_view

    def create_chunks_from_artifact(
        self,
        artifact: dict[str, Any],
        document_id: str | None = None,
    ) -> list[Chunk]:
        """
        从单个文物记录创建多视图文本块

        Args:
            artifact: 文物JSON数据
            document_id: 文档ID，默认使用caseId

        Returns:
            多个视图的Chunk列表
        """
        chunks = []
        case_id = artifact.get("caseId", "")
        doc_id = document_id or case_id

        # 提取公共元数据
        common_metadata = self._extract_common_metadata(artifact)

        chunk_index = 0
        for view_type in self.views:
            content = self._build_view_content(artifact, view_type)
            if not content.strip():
                continue

            chunk = Chunk(
                id=generate_id("chunk", f"{case_id}_{view_type.value}"),
                content=content,
                document_id=doc_id,
                chunk_index=chunk_index,
                view_type=view_type,
                source_fields=self._get_source_fields(view_type),
                **common_metadata,
            )
            chunks.append(chunk)
            chunk_index += 1

        logger.debug(f"Created {len(chunks)} view chunks for artifact {case_id}")
        return chunks

    def create_chunks_from_artifacts(
        self,
        artifacts: list[dict[str, Any]],
    ) -> list[Chunk]:
        """
        批量处理文物记录

        Args:
            artifacts: 文物JSON数据列表

        Returns:
            所有文物的多视图Chunk列表
        """
        all_chunks = []
        for artifact in artifacts:
            chunks = self.create_chunks_from_artifact(artifact)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} total view chunks from {len(artifacts)} artifacts")
        return all_chunks

    def _extract_common_metadata(self, artifact: dict[str, Any]) -> dict[str, Any]:
        """提取公共元数据"""
        # 解析年代
        from kg_search.extraction.structured_extractor import (
            extract_dynasty,
            normalize_year_to_ce,
        )

        year_ce, year_ce_end = normalize_year_to_ce(artifact.get("descAge", ""))
        dynasty = extract_dynasty(artifact.get("descAge", ""))

        # 提取地址信息
        city, district, address, full_address = "", "", "", ""
        addresses = artifact.get("addresses", [])
        if addresses:
            addr = addresses[0]
            city = addr.get("cityName", "")
            district = addr.get("distName", "")
            address = addr.get("address", "")
            full_address = f"{city}{district}{address}"

        # 提取资产类型
        assets_type_code, assets_type_name = "", ""
        asset_types = artifact.get("assetsTypes", [])
        if asset_types:
            atype = asset_types[0]
            assets_type_code = atype.get("subCode", "")
            assets_type_name = atype.get("subName", "")

        # 提取保管信息
        keeper, save_space, save_space_identity = "", "", ""
        keep_depts = artifact.get("keepDepts", [])
        if keep_depts:
            keeper = keep_depts[0].get("name", "")
        keep_places = artifact.get("keepPlaces", [])
        if keep_places:
            place = keep_places[0]
            save_space = place.get("saveSpace", "")
            save_space_identity = place.get("saveSpaceIdentity", "")

        return {
            # 基础标识
            "artifact_id": artifact.get("caseId", ""),
            "artifact_name": artifact.get("caseName", ""),
            # 分类信息
            "assets_classify_code": artifact.get("assetsClassifyCode", ""),
            "assets_classify_name": artifact.get("assetsClassifyName", ""),
            "assets_type_code": assets_type_code,
            "assets_type_name": assets_type_name,
            # 时间信息
            "dynasty": dynasty,
            "age_description": artifact.get("descAge", ""),
            "year_ce": year_ce,
            "year_ce_end": year_ce_end,
            # 地理信息
            "city": city,
            "district": district,
            "address": address,
            "full_address": full_address,
            # 物理属性
            "material": artifact.get("descMaterial", ""),
            "size": artifact.get("descSize", ""),
            "creator": artifact.get("descCreator", ""),
            # 收藏保管
            "keeper": keeper,
            "save_space": save_space,
            "save_space_identity": save_space_identity,
            # 管理信息
            "gov_institution": artifact.get("govInstitution", ""),
            "gov_dept_name": artifact.get("govDeptName", ""),
            "reservation_code": artifact.get("reservationCode", ""),
            # 来源与状态
            "past_history_source": artifact.get("pastHistorySource", ""),
            "reserve_status": artifact.get("reserveStatus", ""),
        }

    def _build_view_content(self, artifact: dict[str, Any], view_type: ViewType) -> str:
        """构建特定视图的文本内容"""
        if view_type == ViewType.FULL:
            return self._build_full_view(artifact)
        elif view_type == ViewType.BASIC:
            return self._build_basic_view(artifact)
        elif view_type == ViewType.DESCRIPTION:
            return self._build_description_view(artifact)
        elif view_type == ViewType.HISTORICAL:
            return self._build_historical_view(artifact)
        elif view_type == ViewType.PHYSICAL:
            return self._build_physical_view(artifact)
        elif view_type == ViewType.LOCATION:
            return self._build_location_view(artifact)
        else:
            return ""

    def _build_full_view(self, artifact: dict[str, Any]) -> str:
        """构建完整视图"""
        sections = []

        # 基础信息
        sections.append(f"【文物名称】{artifact.get('caseName', '')}")
        sections.append(f"【资产等级】{artifact.get('assetsClassifyName', '')}")

        # 资产类型
        asset_types = artifact.get("assetsTypes", [])
        if asset_types:
            type_strs = [f"{t.get('name', '')} - {t.get('subName', '')}" for t in asset_types]
            sections.append(f"【资产类型】{'; '.join(type_strs)}")

        # 时间信息
        if artifact.get("descAge"):
            sections.append(f"【年代】{artifact.get('descAge')}")

        # 物理属性
        if artifact.get("descMaterial"):
            sections.append(f"【材质】{artifact.get('descMaterial')}")
        if artifact.get("descSize"):
            sections.append(f"【尺寸】{artifact.get('descSize')}")
        if artifact.get("descCreator"):
            sections.append(f"【创作者】{artifact.get('descCreator')}")

        # 地理信息
        addresses = artifact.get("addresses", [])
        if addresses:
            addr = addresses[0]
            full_addr = (
                f"{addr.get('cityName', '')}{addr.get('distName', '')}{addr.get('address', '')}"
            )
            sections.append(f"【所在地】{full_addr}")

        # 保管信息
        keep_depts = artifact.get("keepDepts", [])
        if keep_depts:
            keepers = [d.get("name", "") for d in keep_depts]
            sections.append(f"【保管单位】{'; '.join(keepers)}")

        keep_places = artifact.get("keepPlaces", [])
        if keep_places:
            spaces = [p.get("saveSpace", "") for p in keep_places if p.get("saveSpace")]
            if spaces:
                sections.append(f"【存放空间】{'; '.join(spaces)}")

        # 状态与描述
        if artifact.get("reserveStatus"):
            sections.append(f"【保存状态】{artifact.get('reserveStatus')}")
        if artifact.get("reserveStatusDesc"):
            sections.append(f"【详细描述】{artifact.get('reserveStatusDesc')}")

        # 登录信息
        if artifact.get("registerReason"):
            sections.append(f"【登录原因】{artifact.get('registerReason')}")

        # 判定标准
        criteria = artifact.get("judgeCriteria", [])
        if criteria:
            sections.append(f"【判定标准】{'; '.join(criteria)}")

        return "\n".join([s for s in sections if s.split("】")[1].strip() if "】" in s])

    def _build_basic_view(self, artifact: dict[str, Any]) -> str:
        """构建基础信息视图 - 用于快速匹配"""
        sections = []

        sections.append(f"文物: {artifact.get('caseName', '')}")
        sections.append(f"等级: {artifact.get('assetsClassifyName', '')}")

        asset_types = artifact.get("assetsTypes", [])
        if asset_types:
            type_strs = [t.get("subName", "") for t in asset_types if t.get("subName")]
            if type_strs:
                sections.append(f"类型: {', '.join(type_strs)}")

        if artifact.get("descAge"):
            sections.append(f"年代: {artifact.get('descAge')}")

        if artifact.get("descMaterial"):
            sections.append(f"材质: {artifact.get('descMaterial')}")

        if artifact.get("descCreator"):
            sections.append(f"创作者: {artifact.get('descCreator')}")

        return "\n".join(sections)

    def _build_description_view(self, artifact: dict[str, Any]) -> str:
        """构建详细描述视图 - 用于语义检索"""
        sections = []

        sections.append(f"【{artifact.get('caseName', '')}】")

        if artifact.get("reserveStatusDesc"):
            sections.append(artifact.get("reserveStatusDesc"))

        if artifact.get("registerReason"):
            sections.append(f"登录原因: {artifact.get('registerReason')}")

        if artifact.get("notes"):
            sections.append(f"备注: {artifact.get('notes')}")

        return "\n\n".join(sections)

    def _build_historical_view(self, artifact: dict[str, Any]) -> str:
        """构建历史背景视图 - 用于历史相关检索"""
        sections = []

        sections.append(f"文物: {artifact.get('caseName', '')}")

        if artifact.get("descAge"):
            sections.append(f"年代: {artifact.get('descAge')}")

        if artifact.get("pastHistorySource"):
            source_notes = artifact.get("pastHistorySourceNotes", "")
            source_str = artifact.get("pastHistorySource")
            if source_notes:
                source_str += f" ({source_notes})"
            sections.append(f"来源: {source_str}")

        # 从描述中提取历史相关信息
        desc = artifact.get("reserveStatusDesc", "")
        if desc:
            # 提取包含年份或历史事件的句子
            import re

            sentences = re.split(r"[。\n]", desc)
            historical_sentences = []
            for s in sentences:
                if re.search(r"\d{4}年|明治|大正|昭和|民國|清|日治", s):
                    historical_sentences.append(s.strip())
            if historical_sentences:
                sections.append("历史背景: " + "。".join(historical_sentences[:3]))

        # 公告信息
        announcements = artifact.get("announcementList", [])
        if announcements:
            for ann in announcements[:2]:
                if ann.get("registerDate"):
                    sections.append(f"登录日期: {ann.get('registerDate')}")

        return "\n".join(sections)

    def _build_physical_view(self, artifact: dict[str, Any]) -> str:
        """构建物理属性视图 - 用于材质/尺寸相关检索"""
        sections = []

        sections.append(f"文物: {artifact.get('caseName', '')}")

        if artifact.get("descMaterial"):
            sections.append(f"材质: {artifact.get('descMaterial')}")

        if artifact.get("descSize"):
            sections.append(f"尺寸: {artifact.get('descSize')}")

        if artifact.get("descCreator"):
            sections.append(f"创作者/作者: {artifact.get('descCreator')}")

        if artifact.get("amount"):
            sections.append(f"数量: {artifact.get('amount')}")

        if artifact.get("environment"):
            sections.append(f"存放环境: {artifact.get('environment')}")

        # 保存状态
        if artifact.get("reserveStatus"):
            sections.append(f"保存状态: {artifact.get('reserveStatus')}")

        return "\n".join(sections)

    def _build_location_view(self, artifact: dict[str, Any]) -> str:
        """构建地理位置视图 - 用于地点相关检索"""
        sections = []

        sections.append(f"文物: {artifact.get('caseName', '')}")

        # 地址信息
        addresses = artifact.get("addresses", [])
        for addr in addresses:
            city = addr.get("cityName", "")
            district = addr.get("distName", "")
            address = addr.get("address", "")
            if city or district or address:
                sections.append(f"地址: {city}{district}{address}")

        # 保管单位
        keep_depts = artifact.get("keepDepts", [])
        for dept in keep_depts:
            if dept.get("name"):
                sections.append(f"保管单位: {dept.get('name')}")

        # 存放地点
        keep_places = artifact.get("keepPlaces", [])
        for place in keep_places:
            place_info = []
            if place.get("name"):
                place_info.append(place.get("name"))
            if place.get("saveSpace"):
                place_info.append(f"({place.get('saveSpace')})")
            if place.get("address"):
                place_info.append(f"地址: {place.get('address')}")
            if place_info:
                sections.append(f"存放地点: {' '.join(place_info)}")

        # 主管机关
        if artifact.get("govInstitution"):
            sections.append(f"主管机关: {artifact.get('govInstitution')}")

        return "\n".join(sections)

    def _get_source_fields(self, view_type: ViewType) -> list[str]:
        """获取视图对应的源字段列表"""
        field_mapping = {
            ViewType.FULL: [
                "caseName",
                "assetsClassifyName",
                "assetsTypes",
                "descAge",
                "descMaterial",
                "descSize",
                "descCreator",
                "addresses",
                "keepDepts",
                "keepPlaces",
                "reserveStatus",
                "reserveStatusDesc",
                "registerReason",
                "judgeCriteria",
            ],
            ViewType.BASIC: [
                "caseName",
                "assetsClassifyName",
                "assetsTypes",
                "descAge",
                "descMaterial",
                "descCreator",
            ],
            ViewType.DESCRIPTION: [
                "caseName",
                "reserveStatusDesc",
                "registerReason",
                "notes",
            ],
            ViewType.HISTORICAL: [
                "caseName",
                "descAge",
                "pastHistorySource",
                "pastHistorySourceNotes",
                "reserveStatusDesc",
                "announcementList",
            ],
            ViewType.PHYSICAL: [
                "caseName",
                "descMaterial",
                "descSize",
                "descCreator",
                "amount",
                "environment",
                "reserveStatus",
            ],
            ViewType.LOCATION: [
                "caseName",
                "addresses",
                "keepDepts",
                "keepPlaces",
                "govInstitution",
            ],
        }
        return field_mapping.get(view_type, [])


# ============================================================
# 查询视图选择器
# ============================================================


class QueryViewSelector:
    """
    根据查询意图选择最佳视图类型
    """

    # 关键词到视图的映射
    VIEW_KEYWORDS = {
        ViewType.BASIC: [
            "什么",
            "哪个",
            "名称",
            "叫什么",
            "是什么",
        ],
        ViewType.DESCRIPTION: [
            "介绍",
            "描述",
            "详细",
            "说明",
            "讲讲",
            "告诉我关于",
        ],
        ViewType.HISTORICAL: [
            "历史",
            "年代",
            "时期",
            "朝代",
            "什么时候",
            "来源",
            "来历",
            "清代",
            "日治",
            "民国",
            "明治",
            "昭和",
            "同治",
        ],
        ViewType.PHYSICAL: [
            "材质",
            "材料",
            "尺寸",
            "大小",
            "多大",
            "多高",
            "做的",
            "青铜",
            "铜",
            "玉",
            "陶",
            "瓷",
            "纸",
            "木",
            "石",
        ],
        ViewType.LOCATION: [
            "在哪",
            "哪里",
            "地址",
            "位置",
            "收藏",
            "保管",
            "存放",
            "博物馆",
            "哪个馆",
            "台北",
            "台南",
            "高雄",
            "嘉义",
        ],
    }

    @classmethod
    def select_views(cls, query: str) -> list[ViewType]:
        """
        根据查询内容选择合适的视图类型

        Args:
            query: 用户查询

        Returns:
            推荐的视图类型列表（按优先级排序）
        """
        query_lower = query.lower()
        matched_views = []

        for view_type, keywords in cls.VIEW_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    if view_type not in matched_views:
                        matched_views.append(view_type)
                    break

        # 如果没有匹配到特定视图，返回 FULL 和 DESCRIPTION
        if not matched_views:
            matched_views = [ViewType.FULL, ViewType.DESCRIPTION]

        # 总是包含 FULL 作为兜底
        if ViewType.FULL not in matched_views:
            matched_views.append(ViewType.FULL)

        return matched_views

    @classmethod
    def get_view_weights(cls, query: str) -> dict[ViewType, float]:
        """
        获取各视图类型的权重

        Args:
            query: 用户查询

        Returns:
            视图类型到权重的映射
        """
        selected_views = cls.select_views(query)
        weights = {}

        # 主视图权重最高
        for i, view in enumerate(selected_views):
            # 第一个匹配的视图权重1.0，依次递减
            weights[view] = 1.0 - (i * 0.15)

        # 未选中的视图给予较低权重
        for view_type in ViewType:
            if view_type not in weights:
                weights[view_type] = 0.3

        return weights
