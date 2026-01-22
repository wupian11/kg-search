"""
统一的实体和关系类型定义

文博领域知识图谱的标准类型体系
"""

from enum import Enum


class EntityType(str, Enum):
    """
    文博领域实体类型 - 统一标准

    分类:
    - 核心实体: 文物、朝代、年代等基础类型
    - 属性实体: 材质、工艺、尺寸等描述性类型
    - 组织实体: 收藏机构、政府机关等组织类型
    - 扩展实体: 资产等级、判定标准等业务类型
    """

    # === 核心实体 ===
    ARTIFACT = "文物"
    DYNASTY = "朝代"
    PERIOD = "年代"
    CULTURE = "文化"
    EVENT = "事件"

    # === 属性实体 ===
    MATERIAL = "材质"
    TECHNIQUE = "工艺"
    DIMENSION = "尺寸"
    STYLE = "风格"

    # === 空间实体 ===
    LOCATION = "地点"
    STORAGE_SPACE = "存放空间"

    # === 人物实体 ===
    PERSON = "人物"

    # === 组织实体 ===
    MUSEUM = "收藏机构"
    GOV_INSTITUTION = "公告机关"
    ADMIN_AUTHORITY = "主管机关"

    # === 分类实体 ===
    ASSET_LEVEL = "资产等级"
    ASSET_TYPE = "资产类型"

    # === 业务实体 ===
    CRITERIA = "判定标准"
    PROVENANCE = "来源方式"


class RelationType(str, Enum):
    """
    文博领域关系类型 - 统一标准

    分类:
    - 时空关系: 年代归属、地理位置等
    - 属性关系: 材质、创作者等属性关联
    - 机构关系: 保管、登录、主管等组织关联
    - 分类关系: 资产等级、类型归属等
    - 相似关系: 同材质、同年代等相似性关联
    """

    # === 时空关系 ===
    BELONGS_TO_PERIOD = "属于年代"
    BELONGS_TO_DYNASTY = "属于朝代"
    LOCATED_IN = "位于"
    EXCAVATED_FROM = "出土于"

    # === 属性关系 ===
    MADE_OF = "材质为"
    CREATED_BY = "创作者"
    DISCOVERED_BY = "发现者"

    # === 机构关系 ===
    KEPT_BY = "保管于"
    COLLECTED_BY = "收藏于"
    REGISTERED_BY = "登录于"
    ADMINISTERED_BY = "主管为"

    # === 分类关系 ===
    BELONGS_TO_LEVEL = "属于资产等级"
    BELONGS_TO_TYPE = "属于资产类型"
    BELONGS_TO_CULTURE = "属于文化"
    STORED_IN = "存放于"
    MEETS_CRITERIA = "符合标准"
    PROVENANCE_TYPE = "来源方式"

    # === 相似关系 ===
    SAME_PERIOD = "同年代"
    SAME_MATERIAL = "同材质"
    SAME_TYPE = "同类型"
    SAME_REGION = "同区域"
    SAME_KEEPER = "同保管"
    SIMILAR_STYLE = "风格相似"
    TECHNIQUE_INHERIT = "工艺传承"

    # === 通用关系 ===
    RELATED_TO = "相关"


# 类型别名映射（兼容旧代码中的字符串类型）
ENTITY_TYPE_ALIASES: dict[str, EntityType] = {
    # 中文别名
    "文物": EntityType.ARTIFACT,
    "朝代": EntityType.DYNASTY,
    "年代": EntityType.PERIOD,
    "材质": EntityType.MATERIAL,
    "工艺": EntityType.TECHNIQUE,
    "地点": EntityType.LOCATION,
    "收藏机构": EntityType.MUSEUM,
    "人物": EntityType.PERSON,
    "文化": EntityType.CULTURE,
    "尺寸": EntityType.DIMENSION,
    "风格": EntityType.STYLE,
    "事件": EntityType.EVENT,
    # 扩展类型别名
    "资产等级": EntityType.ASSET_LEVEL,
    "资产类型": EntityType.ASSET_TYPE,
    "存放空间": EntityType.STORAGE_SPACE,
    "公告机关": EntityType.GOV_INSTITUTION,
    "主管机关": EntityType.ADMIN_AUTHORITY,
    "判定标准": EntityType.CRITERIA,
    "来源方式": EntityType.PROVENANCE,
    # 保管单位映射到收藏机构
    "保管单位": EntityType.MUSEUM,
}

RELATION_TYPE_ALIASES: dict[str, RelationType] = {
    # 中文别名
    "属于年代": RelationType.BELONGS_TO_PERIOD,
    "属于朝代": RelationType.BELONGS_TO_DYNASTY,
    "位于": RelationType.LOCATED_IN,
    "出土于": RelationType.EXCAVATED_FROM,
    "材质为": RelationType.MADE_OF,
    "创作者": RelationType.CREATED_BY,
    "发现者": RelationType.DISCOVERED_BY,
    "保管于": RelationType.KEPT_BY,
    "收藏于": RelationType.COLLECTED_BY,
    "登录于": RelationType.REGISTERED_BY,
    "主管为": RelationType.ADMINISTERED_BY,
    "属于资产等级": RelationType.BELONGS_TO_LEVEL,
    "属于资产类型": RelationType.BELONGS_TO_TYPE,
    "属于文化": RelationType.BELONGS_TO_CULTURE,
    "存放于": RelationType.STORED_IN,
    "符合标准": RelationType.MEETS_CRITERIA,
    "来源方式": RelationType.PROVENANCE_TYPE,
    "同年代": RelationType.SAME_PERIOD,
    "同材质": RelationType.SAME_MATERIAL,
    "同类型": RelationType.SAME_TYPE,
    "同区域": RelationType.SAME_REGION,
    "同保管": RelationType.SAME_KEEPER,
    "风格相似": RelationType.SIMILAR_STYLE,
    "工艺传承": RelationType.TECHNIQUE_INHERIT,
    "相关": RelationType.RELATED_TO,
    # 旧代码兼容
    "同时期": RelationType.SAME_PERIOD,
    "制作者": RelationType.CREATED_BY,
}


def normalize_entity_type(type_value: str | EntityType) -> EntityType | str:
    """
    标准化实体类型

    Args:
        type_value: 实体类型值（字符串或枚举）

    Returns:
        标准化后的 EntityType，如果无法识别则返回原字符串
    """
    if isinstance(type_value, EntityType):
        return type_value

    # 尝试从别名映射
    if type_value in ENTITY_TYPE_ALIASES:
        return ENTITY_TYPE_ALIASES[type_value]

    # 尝试直接转换
    try:
        return EntityType(type_value)
    except ValueError:
        return type_value


def normalize_relation_type(type_value: str | RelationType) -> RelationType | str:
    """
    标准化关系类型

    Args:
        type_value: 关系类型值（字符串或枚举）

    Returns:
        标准化后的 RelationType，如果无法识别则返回原字符串
    """
    if isinstance(type_value, RelationType):
        return type_value

    # 尝试从别名映射
    if type_value in RELATION_TYPE_ALIASES:
        return RELATION_TYPE_ALIASES[type_value]

    # 尝试直接转换
    try:
        return RelationType(type_value)
    except ValueError:
        return type_value
