"""
相似关系生成器

基于属性规则批量生成文物之间的相似关系

注意：此模块独立于 EntityExtractor，用于批量处理多个文物之间的相似性计算
"""

from typing import Any

from kg_search.utils import generate_id, get_logger

from .models import Entity, Relation
from .types import RelationType
from .utils import normalize_year_to_ce

logger = get_logger(__name__)


class SimilarityRelationGenerator:
    """
    基于属性规则生成相似关系

    用于批量计算多个文物之间的相似性，生成同材质、同年代等关系

    使用示例:
    ```python
    generator = SimilarityRelationGenerator(year_threshold=10)
    relations = generator.generate_similarity_relations(artifacts)
    ```
    """

    def __init__(self, year_threshold: int = 10):
        """
        初始化相似关系生成器

        Args:
            year_threshold: 年代相似的阈值（年），默认10年内视为同年代
        """
        self.year_threshold = year_threshold

    def generate_similarity_relations(
        self,
        artifacts: list[dict[str, Any]],
    ) -> list[Relation]:
        """
        批量生成文物之间的相似关系

        Args:
            artifacts: 文物JSON数据列表

        Returns:
            相似关系列表
        """
        relations = []

        # 按属性分组索引
        by_material: dict[str, list[str]] = {}
        by_type: dict[str, list[str]] = {}
        by_city: dict[str, list[str]] = {}
        by_keeper: dict[str, list[str]] = {}
        year_map: dict[str, int] = {}

        # 文物名称到ID的映射
        name_to_id: dict[str, str] = {}

        # 构建索引
        for artifact in artifacts:
            case_name = artifact.get("caseName", "")
            case_id = artifact.get("caseId", "")
            if not case_name:
                continue

            name_to_id[case_name] = case_id

            # 材质索引
            material = artifact.get("descMaterial", "")
            if material:
                by_material.setdefault(material, []).append(case_name)

            # 类型索引
            for atype in artifact.get("assetsTypes", []):
                sub_code = atype.get("subCode", "")
                if sub_code:
                    by_type.setdefault(sub_code, []).append(case_name)

            # 城市索引
            for addr in artifact.get("addresses", []):
                city = addr.get("cityName", "")
                if city:
                    by_city.setdefault(city, []).append(case_name)

            # 保管单位索引
            for dept in artifact.get("keepDepts", []):
                keeper = dept.get("name", "")
                if keeper:
                    by_keeper.setdefault(keeper, []).append(case_name)

            # 年代索引
            year_ce, _ = normalize_year_to_ce(artifact.get("descAge", ""))
            if year_ce:
                year_map[case_name] = year_ce

        # 生成同材质关系
        relations.extend(
            self._generate_group_relations(by_material, RelationType.SAME_MATERIAL, name_to_id)
        )

        # 生成同类型关系
        relations.extend(
            self._generate_group_relations(by_type, RelationType.SAME_TYPE, name_to_id)
        )

        # 生成同区域关系
        relations.extend(
            self._generate_group_relations(by_city, RelationType.SAME_REGION, name_to_id)
        )

        # 生成同保管关系
        relations.extend(
            self._generate_group_relations(by_keeper, RelationType.SAME_KEEPER, name_to_id)
        )

        # 生成同年代关系
        relations.extend(self._generate_year_similarity_relations(year_map, name_to_id))

        logger.info(
            "Generated similarity relations",
            total=len(relations),
            same_material=len(
                [r for r in relations if r.relation_type == RelationType.SAME_MATERIAL]
            ),
            same_type=len([r for r in relations if r.relation_type == RelationType.SAME_TYPE]),
            same_region=len([r for r in relations if r.relation_type == RelationType.SAME_REGION]),
            same_keeper=len([r for r in relations if r.relation_type == RelationType.SAME_KEEPER]),
            same_period=len([r for r in relations if r.relation_type == RelationType.SAME_PERIOD]),
        )

        return relations

    def generate_from_entities(
        self,
        entities: list[Entity],
    ) -> list[Relation]:
        """
        从实体列表生成相似关系

        Args:
            entities: 实体列表（应为文物类型实体）

        Returns:
            相似关系列表
        """
        relations = []

        # 筛选文物实体
        artifacts = [e for e in entities if e.type_value == "文物"]

        if len(artifacts) < 2:
            return relations

        # 按属性分组
        by_material: dict[str, list[Entity]] = {}
        by_dynasty: dict[str, list[Entity]] = {}

        for artifact in artifacts:
            # 材质分组
            material = artifact.attributes.get("material")
            if material:
                by_material.setdefault(material, []).append(artifact)

            # 朝代分组
            dynasty = artifact.attributes.get("dynasty")
            if dynasty:
                by_dynasty.setdefault(dynasty, []).append(artifact)

        # 生成同材质关系
        for material, members in by_material.items():
            if len(members) < 2:
                continue
            for i, source in enumerate(members):
                for target in members[i + 1 :]:
                    relations.append(
                        Relation.from_entities(
                            source=source,
                            target=target,
                            relation_type=RelationType.SAME_MATERIAL,
                            description=f"{source.name}与{target.name}同材质（{material}）",
                            confidence=0.9,
                        )
                    )

        # 生成同朝代关系
        for dynasty, members in by_dynasty.items():
            if len(members) < 2:
                continue
            for i, source in enumerate(members):
                for target in members[i + 1 :]:
                    relations.append(
                        Relation.from_entities(
                            source=source,
                            target=target,
                            relation_type=RelationType.SAME_PERIOD,
                            description=f"{source.name}与{target.name}同朝代（{dynasty}）",
                            confidence=0.85,
                        )
                    )

        return relations

    def _generate_group_relations(
        self,
        group_map: dict[str, list[str]],
        relation_type: RelationType,
        name_to_id: dict[str, str],
    ) -> list[Relation]:
        """根据分组生成相似关系"""
        relations = []

        for group_key, members in group_map.items():
            if len(members) < 2:
                continue

            # 为组内每对成员创建双向关系
            for i, source in enumerate(members):
                for target in members[i + 1 :]:
                    # A -> B
                    relations.append(
                        Relation(
                            id=generate_id("relation"),
                            source_entity_id=name_to_id.get(source, ""),
                            source_entity_name=source,
                            target_entity_id=name_to_id.get(target, ""),
                            target_entity_name=target,
                            relation_type=relation_type,
                            description=f"{source}与{target}{relation_type.value}（{group_key}）",
                            confidence=0.9,
                        )
                    )
                    # B -> A
                    relations.append(
                        Relation(
                            id=generate_id("relation"),
                            source_entity_id=name_to_id.get(target, ""),
                            source_entity_name=target,
                            target_entity_id=name_to_id.get(source, ""),
                            target_entity_name=source,
                            relation_type=relation_type,
                            description=f"{target}与{source}{relation_type.value}（{group_key}）",
                            confidence=0.9,
                        )
                    )

        return relations

    def _generate_year_similarity_relations(
        self,
        year_map: dict[str, int],
        name_to_id: dict[str, str],
    ) -> list[Relation]:
        """生成同年代关系"""
        relations = []
        names = list(year_map.keys())

        for i, source in enumerate(names):
            source_year = year_map[source]
            for target in names[i + 1 :]:
                target_year = year_map[target]

                if abs(source_year - target_year) <= self.year_threshold:
                    # A -> B
                    relations.append(
                        Relation(
                            id=generate_id("relation"),
                            source_entity_id=name_to_id.get(source, ""),
                            source_entity_name=source,
                            target_entity_id=name_to_id.get(target, ""),
                            target_entity_name=target,
                            relation_type=RelationType.SAME_PERIOD,
                            description=f"{source}({source_year})与{target}({target_year})属于同年代",
                            confidence=0.85,
                        )
                    )
                    # B -> A
                    relations.append(
                        Relation(
                            id=generate_id("relation"),
                            source_entity_id=name_to_id.get(target, ""),
                            source_entity_name=target,
                            target_entity_id=name_to_id.get(source, ""),
                            target_entity_name=source,
                            relation_type=RelationType.SAME_PERIOD,
                            description=f"{target}({target_year})与{source}({source_year})属于同年代",
                            confidence=0.85,
                        )
                    )

        return relations
