"""
统一的实体和关系数据模型

提供标准化的 Entity 和 Relation 数据结构
"""

from dataclasses import dataclass, field
from typing import Any

from kg_search.utils import generate_id

from .types import (
    EntityType,
    RelationType,
    normalize_entity_type,
    normalize_relation_type,
)


@dataclass
class Entity:
    """
    实体数据结构 - 统一标准

    Attributes:
        id: 实体唯一标识
        name: 实体名称
        type: 实体类型（EntityType 枚举或字符串）
        description: 实体描述
        attributes: 扩展属性字典
        source_doc_id: 来源文档ID
        source_chunk_id: 来源文本块ID
        confidence: 置信度 (0.0-1.0)
        name_normalized: 标准化名称（如繁简转换后）
    """

    id: str
    name: str
    type: EntityType | str
    description: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    source_doc_id: str = ""
    source_chunk_id: str = ""
    confidence: float = 1.0
    name_normalized: str = ""

    def __post_init__(self):
        """初始化后处理"""
        # 标准化类型
        self.type = normalize_entity_type(self.type)
        # 如果没有设置标准化名称，使用原名称
        if not self.name_normalized:
            self.name_normalized = self.name

    @property
    def type_value(self) -> str:
        """获取类型的字符串值"""
        if isinstance(self.type, EntityType):
            return self.type.value
        return str(self.type)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type_value,
            "description": self.description,
            "attributes": self.attributes,
            "source_doc_id": self.source_doc_id,
            "source_chunk_id": self.source_chunk_id,
            "confidence": self.confidence,
            "name_normalized": self.name_normalized,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entity":
        """从字典创建实体"""
        return cls(
            id=data.get("id", generate_id("entity")),
            name=data.get("name", ""),
            type=data.get("type", ""),
            description=data.get("description", ""),
            attributes=data.get("attributes", {}),
            source_doc_id=data.get("source_doc_id", ""),
            source_chunk_id=data.get("source_chunk_id", ""),
            confidence=data.get("confidence", 1.0),
            name_normalized=data.get("name_normalized", ""),
        )

    def __hash__(self):
        """支持哈希以便用于集合"""
        return hash((self.name.lower(), self.type_value))

    def __eq__(self, other):
        """相等性比较"""
        if not isinstance(other, Entity):
            return False
        return self.name.lower() == other.name.lower() and self.type_value == other.type_value


@dataclass
class Relation:
    """
    关系数据结构 - 统一标准

    Attributes:
        id: 关系唯一标识
        source_entity_id: 源实体ID
        source_entity_name: 源实体名称
        target_entity_id: 目标实体ID
        target_entity_name: 目标实体名称
        relation_type: 关系类型（RelationType 枚举或字符串）
        description: 关系描述
        attributes: 扩展属性字典
        weight: 关系权重
        confidence: 置信度 (0.0-1.0)
        source_doc_id: 来源文档ID
    """

    id: str
    source_entity_id: str
    source_entity_name: str
    target_entity_id: str
    target_entity_name: str
    relation_type: RelationType | str
    description: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    confidence: float = 1.0
    source_doc_id: str = ""

    def __post_init__(self):
        """初始化后处理"""
        # 标准化类型
        self.relation_type = normalize_relation_type(self.relation_type)

    @property
    def type_value(self) -> str:
        """获取类型的字符串值"""
        if isinstance(self.relation_type, RelationType):
            return self.relation_type.value
        return str(self.relation_type)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "source_entity_id": self.source_entity_id,
            "source_entity_name": self.source_entity_name,
            "target_entity_id": self.target_entity_id,
            "target_entity_name": self.target_entity_name,
            "relation_type": self.type_value,
            "description": self.description,
            "attributes": self.attributes,
            "weight": self.weight,
            "confidence": self.confidence,
            "source_doc_id": self.source_doc_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Relation":
        """从字典创建关系"""
        return cls(
            id=data.get("id", generate_id("relation")),
            source_entity_id=data.get("source_entity_id", ""),
            source_entity_name=data.get("source_entity_name", ""),
            target_entity_id=data.get("target_entity_id", ""),
            target_entity_name=data.get("target_entity_name", ""),
            relation_type=data.get("relation_type", ""),
            description=data.get("description", ""),
            attributes=data.get("attributes", {}),
            weight=data.get("weight", 1.0),
            confidence=data.get("confidence", 1.0),
            source_doc_id=data.get("source_doc_id", ""),
        )

    @classmethod
    def from_entities(
        cls,
        source: Entity,
        target: Entity,
        relation_type: RelationType | str,
        description: str = "",
        confidence: float = 1.0,
        source_doc_id: str = "",
    ) -> "Relation":
        """
        从两个实体创建关系

        Args:
            source: 源实体
            target: 目标实体
            relation_type: 关系类型
            description: 关系描述
            confidence: 置信度
            source_doc_id: 来源文档ID
        """
        return cls(
            id=generate_id("relation"),
            source_entity_id=source.id,
            source_entity_name=source.name,
            target_entity_id=target.id,
            target_entity_name=target.name,
            relation_type=relation_type,
            description=description or f"{source.name} {relation_type} {target.name}",
            confidence=confidence,
            source_doc_id=source_doc_id or source.source_doc_id,
        )

    def __hash__(self):
        """支持哈希以便用于集合"""
        return hash(
            (
                self.source_entity_name.lower(),
                self.target_entity_name.lower(),
                self.type_value,
            )
        )

    def __eq__(self, other):
        """相等性比较"""
        if not isinstance(other, Relation):
            return False
        return (
            self.source_entity_name.lower() == other.source_entity_name.lower()
            and self.target_entity_name.lower() == other.target_entity_name.lower()
            and self.type_value == other.type_value
        )


# === 工具函数 ===


def deduplicate_entities(entities: list[Entity]) -> list[Entity]:
    """
    实体去重

    基于 (name, type) 去重，合并属性

    Args:
        entities: 实体列表

    Returns:
        去重后的实体列表
    """
    seen: dict[tuple[str, str], Entity] = {}

    for entity in entities:
        key = (entity.name.lower(), entity.type_value)
        if key not in seen:
            seen[key] = entity
        else:
            # 合并属性
            existing = seen[key]
            existing.attributes.update(entity.attributes)
            # 保留更高的置信度
            if entity.confidence > existing.confidence:
                existing.confidence = entity.confidence
            # 合并描述
            if entity.description and not existing.description:
                existing.description = entity.description

    return list(seen.values())


def deduplicate_relations(relations: list[Relation]) -> list[Relation]:
    """
    关系去重

    基于 (source_name, target_name, type) 去重

    Args:
        relations: 关系列表

    Returns:
        去重后的关系列表
    """
    seen: dict[tuple[str, str, str], Relation] = {}

    for relation in relations:
        key = (
            relation.source_entity_name.lower(),
            relation.target_entity_name.lower(),
            relation.type_value,
        )
        if key not in seen:
            seen[key] = relation
        else:
            # 保留更高的置信度/权重
            existing = seen[key]
            if relation.confidence > existing.confidence:
                existing.confidence = relation.confidence
            if relation.weight > existing.weight:
                existing.weight = relation.weight

    return list(seen.values())
