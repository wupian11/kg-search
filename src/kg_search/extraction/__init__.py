"""
知识提取模块

统一的实体和关系抽取接口
"""

# 统一类型定义
from .types import (
    EntityType,
    RelationType,
    ENTITY_TYPE_ALIASES,
    RELATION_TYPE_ALIASES,
    normalize_entity_type,
    normalize_relation_type,
)

# 统一数据模型
from .models import (
    Entity,
    Relation,
    deduplicate_entities,
    deduplicate_relations,
)

# 抽取器
from .entity_extractor import EntityExtractor, ExtractorConfig
from .relation_extractor import RelationExtractor
from .similarity_generator import SimilarityRelationGenerator

# 图构建
from .graph_builder import GraphBuilder

# 工具函数
from .utils import (
    normalize_year_to_ce,
    extract_dynasty,
    to_simplified,
    to_traditional,
)

# 兼容旧代码：StructuredExtractor 别名指向 EntityExtractor
# 已废弃，建议直接使用 EntityExtractor.extract_from_json()
StructuredExtractor = EntityExtractor
StructuredExtractorConfig = ExtractorConfig


__all__ = [
    # 类型定义
    "EntityType",
    "RelationType",
    "ENTITY_TYPE_ALIASES",
    "RELATION_TYPE_ALIASES",
    "normalize_entity_type",
    "normalize_relation_type",
    # 数据模型
    "Entity",
    "Relation",
    "deduplicate_entities",
    "deduplicate_relations",
    # 抽取器
    "EntityExtractor",
    "ExtractorConfig",
    "RelationExtractor",
    "SimilarityRelationGenerator",
    # 图构建
    "GraphBuilder",
    # 工具函数
    "normalize_year_to_ce",
    "extract_dynasty",
    "to_simplified",
    "to_traditional",
    # 兼容旧代码（已废弃）
    "StructuredExtractor",
    "StructuredExtractorConfig",
]
