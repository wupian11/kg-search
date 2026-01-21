"""知识提取模块"""

from .entity_extractor import Entity, EntityExtractor, EntityType
from .graph_builder import GraphBuilder
from .relation_extractor import Relation, RelationExtractor
from .structured_extractor import (
    SimilarityRelationGenerator,
    StructuredExtractor,
    StructuredExtractorConfig,
    extract_dynasty,
    normalize_year_to_ce,
    to_simplified,
    to_traditional,
)

__all__ = [
    # 实体抽取
    "EntityExtractor",
    "Entity",
    "EntityType",
    # 关系抽取
    "RelationExtractor",
    "Relation",
    # 图构建
    "GraphBuilder",
    # 结构化抽取
    "StructuredExtractor",
    "StructuredExtractorConfig",
    "SimilarityRelationGenerator",
    # 工具函数
    "normalize_year_to_ce",
    "extract_dynasty",
    "to_simplified",
    "to_traditional",
]
