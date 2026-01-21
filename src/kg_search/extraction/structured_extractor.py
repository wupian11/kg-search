"""
结构化字段抽取器

直接从JSON结构化数据映射实体和关系，支持台湾文物数据格式
结合 LLM 补充从描述性文本中抽取隐含实体
"""

import re
from dataclasses import dataclass, field
from typing import Any

from kg_search.utils import generate_id, get_logger

from .entity_extractor import Entity, EntityType
from .relation_extractor import Relation

logger = get_logger(__name__)


# ============================================================
# 年代标准化工具
# ============================================================

# 日治时期年号转西元（基准年）
JAPANESE_ERA_BASE = {
    "明治": 1868,
    "大正": 1912,
    "昭和": 1926,
    "平成": 1989,
    "令和": 2019,
}

# 民国纪年基准
MINGUO_BASE = 1911


def normalize_year_to_ce(age_str: str) -> tuple[int | None, int | None]:
    """
    将各种年代格式标准化为西元年份

    支持格式:
    - 西元1932年, 公元前1600年
    - 日治明治39年, 日治昭和7年
    - 民國65年
    - 清同治九年(1870)
    - 1949

    Returns:
        (year_start, year_end) - 年份范围，单一年份时 year_end 为 None
    """
    if not age_str:
        return None, None

    # 清理字符串
    age_str = age_str.strip()

    # 1. 尝试匹配带括号的西元年份 如 "清同治九年(1870)"
    bracket_match = re.search(r"\((\d{3,4})\)", age_str)
    if bracket_match:
        return int(bracket_match.group(1)), None

    # 2. 尝试匹配西元纪年 如 "西元1932年"
    ce_match = re.search(r"西元\s*(\d{3,4})\s*年", age_str)
    if ce_match:
        return int(ce_match.group(1)), None

    # 3. 尝试匹配公元纪年（含公元前）
    ce_match2 = re.search(r"公元(前)?(\d{3,4})年", age_str)
    if ce_match2:
        year = int(ce_match2.group(2))
        if ce_match2.group(1):  # 公元前
            return -year, None
        return year, None

    # 4. 尝试匹配日治时期年号
    jp_era_match = re.search(r"(明治|大正|昭和|平成|令和)\s*(\d{1,2})\s*年", age_str)
    if jp_era_match:
        era = jp_era_match.group(1)
        year_num = int(jp_era_match.group(2))
        if era in JAPANESE_ERA_BASE:
            # 日本年号第1年对应基准年
            return JAPANESE_ERA_BASE[era] + year_num - 1, None

    # 5. 尝试匹配民国纪年
    minguo_match = re.search(r"民[國国]\s*(\d{1,3})\s*年", age_str)
    if minguo_match:
        year_num = int(minguo_match.group(1))
        return MINGUO_BASE + year_num, None

    # 6. 尝试匹配纯数字年份
    pure_year_match = re.search(r"^(\d{4})$", age_str.strip())
    if pure_year_match:
        return int(pure_year_match.group(1)), None

    # 7. 尝试匹配年份范围 如 "1906-1932"
    range_match = re.search(r"(\d{4})\s*[-~至到]\s*(\d{4})", age_str)
    if range_match:
        return int(range_match.group(1)), int(range_match.group(2))

    # 无法解析
    logger.debug(f"无法解析年代: {age_str}")
    return None, None


def extract_dynasty(age_str: str) -> str | None:
    """
    从年代描述中提取朝代/时期
    """
    if not age_str:
        return None

    # 清代
    if re.search(r"清|同治|光緒|宣統|道光|咸豐|嘉慶|乾隆", age_str):
        return "清代"

    # 日治时期
    if re.search(r"日治|明治|大正|昭和", age_str):
        return "日治時期"

    # 民国
    if re.search(r"民[國国]", age_str):
        return "民國"

    return None


# ============================================================
# 繁简转换工具
# ============================================================

# 常用繁简转换映射（可扩展）
TRAD_TO_SIMP = {
    "臺": "台",
    "灣": "湾",
    "國": "国",
    "區": "区",
    "歷": "历",
    "藝": "艺",
    "術": "术",
    "價": "价",
    "圖": "图",
    "書": "书",
    "館": "馆",
    "寶": "宝",
    "發": "发",
    "現": "现",
    "鑄": "铸",
    "造": "造",
    "銅": "铜",
    "紙": "纸",
    "質": "质",
    "機": "机",
    "構": "构",
    "單": "单",
    "資": "资",
    "產": "产",
    "類": "类",
    "時": "时",
    "視": "视",
    "廳": "厅",
    "縣": "县",
    "處": "处",
    "學": "学",
    "會": "会",
}

SIMP_TO_TRAD = {v: k for k, v in TRAD_TO_SIMP.items()}


def to_simplified(text: str) -> str:
    """繁体转简体"""
    if not text:
        return text
    result = text
    for trad, simp in TRAD_TO_SIMP.items():
        result = result.replace(trad, simp)
    return result


def to_traditional(text: str) -> str:
    """简体转繁体"""
    if not text:
        return text
    result = text
    for simp, trad in SIMP_TO_TRAD.items():
        result = result.replace(simp, trad)
    return result


# ============================================================
# 结构化抽取器
# ============================================================


@dataclass
class StructuredExtractorConfig:
    """结构化抽取器配置"""

    # 是否自动生成相似关系
    generate_similarity_relations: bool = True

    # 相似关系的年代差异阈值（年）
    year_similarity_threshold: int = 10

    # 是否进行繁简转换
    normalize_to_traditional: bool = True

    # LLM补充抽取的描述字段
    description_fields: list[str] = field(
        default_factory=lambda: [
            "registerReason",
            "reserveStatusDesc",
            "notes",
        ]
    )


class StructuredExtractor:
    """
    结构化字段抽取器

    从台湾文物JSON数据中直接映射实体和关系
    """

    def __init__(self, config: StructuredExtractorConfig | None = None):
        self.config = config or StructuredExtractorConfig()

    def extract_entities_from_artifact(
        self,
        artifact: dict[str, Any],
    ) -> list[Entity]:
        """
        从单个文物记录中抽取实体

        Args:
            artifact: 文物JSON数据

        Returns:
            实体列表
        """
        entities = []
        case_id = artifact.get("caseId", "")

        # 1. 文物实体（核心）
        artifact_entity = self._extract_artifact_entity(artifact)
        if artifact_entity:
            entities.append(artifact_entity)

        # 2. 年代实体
        age_entity = self._extract_age_entity(artifact, case_id)
        if age_entity:
            entities.append(age_entity)

        # 3. 材质实体
        material_entity = self._extract_material_entity(artifact, case_id)
        if material_entity:
            entities.append(material_entity)

        # 4. 行政区实体
        location_entities = self._extract_location_entities(artifact, case_id)
        entities.extend(location_entities)

        # 5. 资产类型实体
        type_entities = self._extract_asset_type_entities(artifact, case_id)
        entities.extend(type_entities)

        # 6. 保管单位实体
        keeper_entities = self._extract_keeper_entities(artifact, case_id)
        entities.extend(keeper_entities)

        # 7. 存放空间实体
        space_entities = self._extract_space_entities(artifact, case_id)
        entities.extend(space_entities)

        # 8. 人物实体（创作者等）
        person_entities = self._extract_person_entities(artifact, case_id)
        entities.extend(person_entities)

        # 9. 公告机关实体
        gov_entities = self._extract_gov_entities(artifact, case_id)
        entities.extend(gov_entities)

        # 10. 判定标准实体
        criteria_entities = self._extract_criteria_entities(artifact, case_id)
        entities.extend(criteria_entities)

        # 11. 来源方式实体
        source_entity = self._extract_source_entity(artifact, case_id)
        if source_entity:
            entities.append(source_entity)

        return entities

    def extract_relations_from_artifact(
        self,
        artifact: dict[str, Any],
        entities: list[Entity],
    ) -> list[Relation]:
        """
        从单个文物记录中抽取关系

        Args:
            artifact: 文物JSON数据
            entities: 已抽取的实体列表

        Returns:
            关系列表
        """
        relations = []
        case_name = artifact.get("caseName", "")

        # 构建实体名称到实体的映射
        entity_map = {e.name: e for e in entities}

        # 1. 文物 - 年代关系
        age_relations = self._extract_age_relations(artifact, case_name, entity_map)
        relations.extend(age_relations)

        # 2. 文物 - 材质关系
        material_relation = self._extract_material_relation(artifact, case_name, entity_map)
        if material_relation:
            relations.append(material_relation)

        # 3. 文物 - 位置关系
        location_relations = self._extract_location_relations(artifact, case_name, entity_map)
        relations.extend(location_relations)

        # 4. 文物 - 资产类型关系
        type_relations = self._extract_type_relations(artifact, case_name, entity_map)
        relations.extend(type_relations)

        # 5. 文物 - 保管单位关系
        keeper_relations = self._extract_keeper_relations(artifact, case_name, entity_map)
        relations.extend(keeper_relations)

        # 6. 文物 - 存放空间关系
        space_relations = self._extract_space_relations(artifact, case_name, entity_map)
        relations.extend(space_relations)

        # 7. 文物 - 创作者关系
        creator_relations = self._extract_creator_relations(artifact, case_name, entity_map)
        relations.extend(creator_relations)

        # 8. 文物 - 公告机关关系
        gov_relations = self._extract_gov_relations(artifact, case_name, entity_map)
        relations.extend(gov_relations)

        # 9. 文物 - 判定标准关系
        criteria_relations = self._extract_criteria_relations(artifact, case_name, entity_map)
        relations.extend(criteria_relations)

        # 10. 文物 - 来源方式关系
        source_relation = self._extract_source_relation(artifact, case_name, entity_map)
        if source_relation:
            relations.append(source_relation)

        return relations

    def get_description_text(self, artifact: dict[str, Any]) -> str:
        """
        获取需要LLM补充抽取的描述性文本

        Args:
            artifact: 文物JSON数据

        Returns:
            拼接的描述文本
        """
        texts = []
        for field_name in self.config.description_fields:
            value = artifact.get(field_name, "")
            if value:
                texts.append(f"【{field_name}】{value}")
        return "\n\n".join(texts)

    # ============================================================
    # 私有方法 - 实体抽取
    # ============================================================

    def _extract_artifact_entity(self, artifact: dict[str, Any]) -> Entity | None:
        """抽取文物实体"""
        case_id = artifact.get("caseId", "")
        case_name = artifact.get("caseName", "")

        if not case_name:
            return None

        # 标准化名称
        name_normalized = (
            to_traditional(case_name) if self.config.normalize_to_traditional else case_name
        )

        return Entity(
            id=generate_id("entity", case_id),
            name=case_name,
            type=EntityType.ARTIFACT,
            description=artifact.get("reserveStatusDesc", "")[:200],
            attributes={
                "case_id": case_id,
                "name_normalized": name_normalized,
                "assets_classify_name": artifact.get("assetsClassifyName", ""),
            },
            source_doc_id=case_id,
            confidence=1.0,
        )

    def _extract_age_entity(self, artifact: dict[str, Any], case_id: str) -> Entity | None:
        """抽取年代实体"""
        desc_age = artifact.get("descAge", "")
        if not desc_age:
            return None

        year_ce, year_ce_end = normalize_year_to_ce(desc_age)
        dynasty = extract_dynasty(desc_age)

        return Entity(
            id=generate_id("entity", f"{case_id}_age"),
            name=desc_age,
            type=EntityType.PERIOD,
            description=f"年代: {desc_age}",
            attributes={
                "year_ce": year_ce,
                "year_ce_end": year_ce_end,
                "dynasty": dynasty,
                "original": desc_age,
            },
            source_doc_id=case_id,
            confidence=1.0,
        )

    def _extract_material_entity(self, artifact: dict[str, Any], case_id: str) -> Entity | None:
        """抽取材质实体"""
        material = artifact.get("descMaterial", "")
        if not material:
            return None

        return Entity(
            id=generate_id("entity", f"{case_id}_material"),
            name=material,
            type=EntityType.MATERIAL,
            description=f"材质: {material}",
            attributes={},
            source_doc_id=case_id,
            confidence=1.0,
        )

    def _extract_location_entities(self, artifact: dict[str, Any], case_id: str) -> list[Entity]:
        """抽取行政区实体"""
        entities = []
        addresses = artifact.get("addresses", [])

        for addr in addresses:
            city = addr.get("cityName", "")
            district = addr.get("distName", "")

            if city:
                # 城市实体
                entities.append(
                    Entity(
                        id=generate_id("entity", f"{case_id}_city_{city}"),
                        name=city,
                        type=EntityType.LOCATION,
                        description=f"城市: {city}",
                        attributes={"level": "city"},
                        source_doc_id=case_id,
                        confidence=1.0,
                    )
                )

            if city and district:
                # 行政区实体
                full_district = f"{city}{district}"
                entities.append(
                    Entity(
                        id=generate_id("entity", f"{case_id}_district_{full_district}"),
                        name=full_district,
                        type=EntityType.LOCATION,
                        description=f"行政区: {full_district}",
                        attributes={"level": "district", "city": city, "district": district},
                        source_doc_id=case_id,
                        confidence=1.0,
                    )
                )

        return entities

    def _extract_asset_type_entities(self, artifact: dict[str, Any], case_id: str) -> list[Entity]:
        """抽取资产类型实体"""
        entities = []

        # 资产分级
        classify_name = artifact.get("assetsClassifyName", "")
        if classify_name:
            entities.append(
                Entity(
                    id=generate_id("entity", f"{case_id}_classify_{classify_name}"),
                    name=classify_name,
                    type="资产等级",
                    description=f"资产分级: {classify_name}",
                    attributes={"code": artifact.get("assetsClassifyCode", "")},
                    source_doc_id=case_id,
                    confidence=1.0,
                )
            )

        # 资产类型
        asset_types = artifact.get("assetsTypes", [])
        for atype in asset_types:
            type_name = atype.get("name", "")
            sub_name = atype.get("subName", "")

            if type_name:
                entities.append(
                    Entity(
                        id=generate_id("entity", f"{case_id}_type_{type_name}"),
                        name=type_name,
                        type="资产类型",
                        description=f"资产类型: {type_name}",
                        attributes={"code": atype.get("code", "")},
                        source_doc_id=case_id,
                        confidence=1.0,
                    )
                )

            if sub_name:
                entities.append(
                    Entity(
                        id=generate_id("entity", f"{case_id}_subtype_{sub_name}"),
                        name=sub_name,
                        type="资产类型",
                        description=f"资产子类型: {sub_name}",
                        attributes={
                            "code": atype.get("subCode", ""),
                            "parent_type": type_name,
                        },
                        source_doc_id=case_id,
                        confidence=1.0,
                    )
                )

        return entities

    def _extract_keeper_entities(self, artifact: dict[str, Any], case_id: str) -> list[Entity]:
        """抽取保管单位实体"""
        entities = []
        keep_depts = artifact.get("keepDepts", [])

        for dept in keep_depts:
            name = dept.get("name", "")
            if name:
                entities.append(
                    Entity(
                        id=generate_id("entity", f"{case_id}_keeper_{name}"),
                        name=name,
                        type=EntityType.MUSEUM,
                        description=f"保管单位: {name}",
                        attributes={"role": "keeper"},
                        source_doc_id=case_id,
                        confidence=1.0,
                    )
                )

        return entities

    def _extract_space_entities(self, artifact: dict[str, Any], case_id: str) -> list[Entity]:
        """抽取存放空间实体"""
        entities = []
        keep_places = artifact.get("keepPlaces", [])

        for place in keep_places:
            save_space = place.get("saveSpace", "")
            if save_space:
                entities.append(
                    Entity(
                        id=generate_id("entity", f"{case_id}_space_{save_space}"),
                        name=save_space,
                        type="存放空间",
                        description=f"存放空间类型: {save_space}",
                        attributes={
                            "identity": place.get("saveSpaceIdentity", ""),
                        },
                        source_doc_id=case_id,
                        confidence=1.0,
                    )
                )

        return entities

    def _extract_person_entities(self, artifact: dict[str, Any], case_id: str) -> list[Entity]:
        """抽取人物实体"""
        entities = []

        # 创作者
        creator = artifact.get("descCreator", "")
        if creator:
            entities.append(
                Entity(
                    id=generate_id("entity", f"{case_id}_creator_{creator}"),
                    name=creator,
                    type=EntityType.PERSON,
                    description=f"创作者: {creator}",
                    attributes={"role": "creator"},
                    source_doc_id=case_id,
                    confidence=1.0,
                )
            )

        return entities

    def _extract_gov_entities(self, artifact: dict[str, Any], case_id: str) -> list[Entity]:
        """抽取政府机关实体"""
        entities = []

        # 公告机关
        gov_name = artifact.get("govInstitutionName", "")
        if gov_name:
            entities.append(
                Entity(
                    id=generate_id("entity", f"{case_id}_gov_{gov_name}"),
                    name=gov_name,
                    type="公告机关",
                    description=f"公告机关: {gov_name}",
                    attributes={},
                    source_doc_id=case_id,
                    confidence=1.0,
                )
            )

        # 主管机关
        gov_inst = artifact.get("govInstitution", "")
        if gov_inst and gov_inst != gov_name:
            entities.append(
                Entity(
                    id=generate_id("entity", f"{case_id}_inst_{gov_inst}"),
                    name=gov_inst,
                    type="主管机关",
                    description=f"主管机关: {gov_inst}",
                    attributes={
                        "dept": artifact.get("govDeptName", ""),
                    },
                    source_doc_id=case_id,
                    confidence=1.0,
                )
            )

        return entities

    def _extract_criteria_entities(self, artifact: dict[str, Any], case_id: str) -> list[Entity]:
        """抽取判定标准实体"""
        entities = []
        criteria_list = artifact.get("judgeCriteria", [])

        for criteria in criteria_list:
            if criteria:
                entities.append(
                    Entity(
                        id=generate_id("entity", f"{case_id}_criteria_{criteria[:20]}"),
                        name=criteria,
                        type="判定标准",
                        description=criteria,
                        attributes={},
                        source_doc_id=case_id,
                        confidence=1.0,
                    )
                )

        return entities

    def _extract_source_entity(self, artifact: dict[str, Any], case_id: str) -> Entity | None:
        """抽取来源方式实体"""
        source = artifact.get("pastHistorySource", "")
        if not source:
            return None

        return Entity(
            id=generate_id("entity", f"{case_id}_source_{source}"),
            name=source,
            type="来源方式",
            description=f"来源方式: {source}",
            attributes={
                "notes": artifact.get("pastHistorySourceNotes", ""),
            },
            source_doc_id=case_id,
            confidence=1.0,
        )

    # ============================================================
    # 私有方法 - 关系抽取
    # ============================================================

    def _extract_age_relations(
        self, artifact: dict[str, Any], case_name: str, entity_map: dict[str, Entity]
    ) -> list[Relation]:
        """抽取年代关系"""
        relations = []
        desc_age = artifact.get("descAge", "")

        if desc_age and desc_age in entity_map:
            relations.append(
                Relation(
                    id=generate_id("relation"),
                    source_entity=case_name,
                    target_entity=desc_age,
                    relation_type="属于年代",
                    description=f"{case_name}的年代为{desc_age}",
                    source_doc_id=artifact.get("caseId", ""),
                    confidence=1.0,
                )
            )

        return relations

    def _extract_material_relation(
        self, artifact: dict[str, Any], case_name: str, entity_map: dict[str, Entity]
    ) -> Relation | None:
        """抽取材质关系"""
        material = artifact.get("descMaterial", "")

        if material and material in entity_map:
            return Relation(
                id=generate_id("relation"),
                source_entity=case_name,
                target_entity=material,
                relation_type="材质为",
                description=f"{case_name}的材质为{material}",
                source_doc_id=artifact.get("caseId", ""),
                confidence=1.0,
            )
        return None

    def _extract_location_relations(
        self, artifact: dict[str, Any], case_name: str, entity_map: dict[str, Entity]
    ) -> list[Relation]:
        """抽取位置关系"""
        relations = []
        addresses = artifact.get("addresses", [])

        for addr in addresses:
            city = addr.get("cityName", "")
            district = addr.get("distName", "")

            if city and district:
                full_district = f"{city}{district}"
                if full_district in entity_map:
                    relations.append(
                        Relation(
                            id=generate_id("relation"),
                            source_entity=case_name,
                            target_entity=full_district,
                            relation_type="位于",
                            description=f"{case_name}位于{full_district}",
                            source_doc_id=artifact.get("caseId", ""),
                            confidence=1.0,
                        )
                    )

        return relations

    def _extract_type_relations(
        self, artifact: dict[str, Any], case_name: str, entity_map: dict[str, Entity]
    ) -> list[Relation]:
        """抽取资产类型关系"""
        relations = []

        # 资产分级关系
        classify_name = artifact.get("assetsClassifyName", "")
        if classify_name and classify_name in entity_map:
            relations.append(
                Relation(
                    id=generate_id("relation"),
                    source_entity=case_name,
                    target_entity=classify_name,
                    relation_type="属于资产等级",
                    description=f"{case_name}属于{classify_name}",
                    source_doc_id=artifact.get("caseId", ""),
                    confidence=1.0,
                )
            )

        # 资产类型关系
        asset_types = artifact.get("assetsTypes", [])
        for atype in asset_types:
            sub_name = atype.get("subName", "")
            if sub_name and sub_name in entity_map:
                relations.append(
                    Relation(
                        id=generate_id("relation"),
                        source_entity=case_name,
                        target_entity=sub_name,
                        relation_type="属于资产类型",
                        description=f"{case_name}属于{sub_name}",
                        source_doc_id=artifact.get("caseId", ""),
                        confidence=1.0,
                    )
                )

        return relations

    def _extract_keeper_relations(
        self, artifact: dict[str, Any], case_name: str, entity_map: dict[str, Entity]
    ) -> list[Relation]:
        """抽取保管关系"""
        relations = []
        keep_depts = artifact.get("keepDepts", [])

        for dept in keep_depts:
            name = dept.get("name", "")
            if name and name in entity_map:
                relations.append(
                    Relation(
                        id=generate_id("relation"),
                        source_entity=case_name,
                        target_entity=name,
                        relation_type="保管于",
                        description=f"{case_name}由{name}保管",
                        source_doc_id=artifact.get("caseId", ""),
                        confidence=1.0,
                    )
                )

        return relations

    def _extract_space_relations(
        self, artifact: dict[str, Any], case_name: str, entity_map: dict[str, Entity]
    ) -> list[Relation]:
        """抽取存放空间关系"""
        relations = []
        keep_places = artifact.get("keepPlaces", [])

        for place in keep_places:
            save_space = place.get("saveSpace", "")
            if save_space and save_space in entity_map:
                relations.append(
                    Relation(
                        id=generate_id("relation"),
                        source_entity=case_name,
                        target_entity=save_space,
                        relation_type="存放于",
                        description=f"{case_name}存放于{save_space}",
                        source_doc_id=artifact.get("caseId", ""),
                        confidence=1.0,
                    )
                )

        return relations

    def _extract_creator_relations(
        self, artifact: dict[str, Any], case_name: str, entity_map: dict[str, Entity]
    ) -> list[Relation]:
        """抽取创作者关系"""
        relations = []
        creator = artifact.get("descCreator", "")

        if creator and creator in entity_map:
            relations.append(
                Relation(
                    id=generate_id("relation"),
                    source_entity=case_name,
                    target_entity=creator,
                    relation_type="创作者",
                    description=f"{case_name}的创作者是{creator}",
                    source_doc_id=artifact.get("caseId", ""),
                    confidence=1.0,
                )
            )

        return relations

    def _extract_gov_relations(
        self, artifact: dict[str, Any], case_name: str, entity_map: dict[str, Entity]
    ) -> list[Relation]:
        """抽取政府机关关系"""
        relations = []

        # 公告机关
        gov_name = artifact.get("govInstitutionName", "")
        if gov_name and gov_name in entity_map:
            relations.append(
                Relation(
                    id=generate_id("relation"),
                    source_entity=case_name,
                    target_entity=gov_name,
                    relation_type="登录于",
                    description=f"{case_name}由{gov_name}登录",
                    source_doc_id=artifact.get("caseId", ""),
                    confidence=1.0,
                )
            )

        # 主管机关
        gov_inst = artifact.get("govInstitution", "")
        if gov_inst and gov_inst in entity_map and gov_inst != gov_name:
            relations.append(
                Relation(
                    id=generate_id("relation"),
                    source_entity=case_name,
                    target_entity=gov_inst,
                    relation_type="主管为",
                    description=f"{case_name}的主管机关是{gov_inst}",
                    source_doc_id=artifact.get("caseId", ""),
                    confidence=1.0,
                )
            )

        return relations

    def _extract_criteria_relations(
        self, artifact: dict[str, Any], case_name: str, entity_map: dict[str, Entity]
    ) -> list[Relation]:
        """抽取判定标准关系"""
        relations = []
        criteria_list = artifact.get("judgeCriteria", [])

        for criteria in criteria_list:
            if criteria and criteria in entity_map:
                relations.append(
                    Relation(
                        id=generate_id("relation"),
                        source_entity=case_name,
                        target_entity=criteria,
                        relation_type="符合标准",
                        description=f"{case_name}符合标准: {criteria}",
                        source_doc_id=artifact.get("caseId", ""),
                        confidence=1.0,
                    )
                )

        return relations

    def _extract_source_relation(
        self, artifact: dict[str, Any], case_name: str, entity_map: dict[str, Entity]
    ) -> Relation | None:
        """抽取来源方式关系"""
        source = artifact.get("pastHistorySource", "")

        if source and source in entity_map:
            return Relation(
                id=generate_id("relation"),
                source_entity=case_name,
                target_entity=source,
                relation_type="来源方式",
                description=f"{case_name}的来源方式为{source}",
                source_doc_id=artifact.get("caseId", ""),
                confidence=1.0,
            )
        return None


# ============================================================
# 相似关系生成器
# ============================================================


class SimilarityRelationGenerator:
    """
    基于属性规则生成相似关系
    """

    def __init__(self, year_threshold: int = 10):
        self.year_threshold = year_threshold

    def generate_similarity_relations(
        self,
        artifacts: list[dict[str, Any]],
    ) -> list[Relation]:
        """
        批量生成文物之间的相似关系

        Args:
            artifacts: 文物列表

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

        # 构建索引
        for artifact in artifacts:
            case_name = artifact.get("caseName", "")
            if not case_name:
                continue

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
        relations.extend(self._generate_group_relations(by_material, "同材质"))

        # 生成同类型关系
        relations.extend(self._generate_group_relations(by_type, "同类型"))

        # 生成同区域关系
        relations.extend(self._generate_group_relations(by_city, "同区域"))

        # 生成同保管关系
        relations.extend(self._generate_group_relations(by_keeper, "同保管"))

        # 生成同年代关系
        relations.extend(self._generate_year_similarity_relations(year_map))

        return relations

    def _generate_group_relations(
        self,
        group_map: dict[str, list[str]],
        relation_type: str,
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
                            source_entity=source,
                            target_entity=target,
                            relation_type=relation_type,
                            description=f"{source}与{target}{relation_type}（{group_key}）",
                            confidence=0.9,
                        )
                    )
                    # B -> A
                    relations.append(
                        Relation(
                            id=generate_id("relation"),
                            source_entity=target,
                            target_entity=source,
                            relation_type=relation_type,
                            description=f"{target}与{source}{relation_type}（{group_key}）",
                            confidence=0.9,
                        )
                    )

        return relations

    def _generate_year_similarity_relations(
        self,
        year_map: dict[str, int],
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
                            source_entity=source,
                            target_entity=target,
                            relation_type="同年代",
                            description=f"{source}({source_year})与{target}({target_year})属于同年代",
                            confidence=0.85,
                        )
                    )
                    # B -> A
                    relations.append(
                        Relation(
                            id=generate_id("relation"),
                            source_entity=target,
                            target_entity=source,
                            relation_type="同年代",
                            description=f"{target}({target_year})与{source}({source_year})属于同年代",
                            confidence=0.85,
                        )
                    )

        return relations
