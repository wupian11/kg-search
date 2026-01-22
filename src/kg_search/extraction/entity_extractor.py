"""
统一实体抽取器

支持多格式输入:
- 文本/Markdown 格式 (LLM抽取)
- JSON 结构化格式 (规则抽取)
- Document 对象 (混合抽取)
"""

import json
from dataclasses import dataclass, field
from typing import Any

from kg_search.ingestion.chunkers.base import Chunk
from kg_search.ingestion.loaders.base import Document
from kg_search.utils import generate_hash_id, generate_id, get_logger

from .models import Entity, Relation, deduplicate_entities, deduplicate_relations
from .prompts.entity_prompt import ENTITY_EXTRACTION_PROMPT, ENTITY_TYPES
from .types import EntityType, RelationType
from .utils import extract_dynasty, normalize_year_to_ce, to_traditional

logger = get_logger(__name__)


@dataclass
class ExtractorConfig:
    """抽取器配置"""

    # 是否启用 LLM 抽取
    enable_llm: bool = True

    # 是否进行繁简转换
    normalize_to_traditional: bool = True

    # LLM 补充抽取的描述字段（用于 JSON 输入）
    description_fields: list[str] = field(
        default_factory=lambda: [
            "registerReason",
            "reserveStatusDesc",
            "notes",
        ]
    )


class EntityExtractor:
    """
    统一实体抽取器

    支持:
    - 文本/Markdown 格式 (LLM抽取)
    - JSON 结构化格式 (规则抽取)
    - Document 对象 (混合抽取)

    使用示例:
    ```python
    # 从文本抽取（需要 LLM）
    extractor = EntityExtractor(llm_client=client)
    entities, relations = await extractor.extract_from_text(md)

    # 从 JSON 抽取（不需要 LLM）
    extractor = EntityExtractor()
    entities, relations = extractor.extract_from_json(json)

    # 从 Document 抽取
    entities, relations = await extractor.extract_from_document(doc)
    ```
    """

    def __init__(
        self,
        llm_client: Any = None,
        config: ExtractorConfig | None = None,
    ):
        """
        初始化实体抽取器

        Args:
            llm_client: LLM客户端实例（用于文本抽取）
            config: 抽取器配置
        """
        self.llm_client = llm_client
        self.config = config or ExtractorConfig()

    @property
    def enable_llm(self) -> bool:
        """是否启用 LLM 抽取"""
        return self.config.enable_llm and self.llm_client is not None

    # ============================================================
    # 公共 API - 统一入口
    # ============================================================

    async def extract(
        self,
        data: str | dict | Document,
        source_id: str = "",
    ) -> tuple[list[Entity], list[Relation]]:
        """
        统一抽取入口 - 自动识别输入格式

        Args:
            data: 文本字符串、JSON字典或Document对象
            source_id: 来源文档ID

        Returns:
            (实体列表, 关系列表)
        """
        if isinstance(data, str):
            return await self.extract_from_text(data, source_id)
        elif isinstance(data, dict):
            return self.extract_from_json(data, source_id)
        elif isinstance(data, Document):
            return await self.extract_from_document(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    # ============================================================
    # 文本抽取 (LLM)
    # ============================================================

    async def extract_from_text(
        self,
        text: str,
        source_doc_id: str = "",
        source_chunk_id: str = "",
    ) -> tuple[list[Entity], list[Relation]]:
        """
        从文本中提取实体（使用LLM）

        Args:
            text: 输入文本
            source_doc_id: 来源文档ID
            source_chunk_id: 来源块ID

        Returns:
            (实体列表, 关系列表)
        """
        if not self.enable_llm:
            logger.warning("LLM not enabled, returning empty results for text extraction")
            return [], []

        entities = await self._llm_extract_entities(text, source_doc_id, source_chunk_id)
        # TODO: 关系抽取可以单独调用 RelationExtractor
        return entities, []

    async def extract_from_chunk(self, chunk: Chunk) -> tuple[list[Entity], list[Relation]]:
        """从文本块中提取实体"""
        return await self.extract_from_text(
            chunk.content,
            source_doc_id=chunk.document_id,
            source_chunk_id=chunk.id,
        )

    async def _llm_extract_entities(
        self,
        text: str,
        source_doc_id: str = "",
        source_chunk_id: str = "",
    ) -> list[Entity]:
        """使用 LLM 从文本中提取实体"""
        prompt = ENTITY_EXTRACTION_PROMPT.format(
            entity_types=ENTITY_TYPES,
            text=text,
        )

        try:
            response = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            # 解析响应
            result = json.loads(response)
            entities_data = result.get("entities", [])

            entities = []
            for data in entities_data:
                entity = Entity.from_dict(data)
                entity.source_doc_id = source_doc_id
                entity.source_chunk_id = source_chunk_id
                if not entity.id:
                    entity.id = generate_id(f"entity_{entity.type_value}")
                entities.append(entity)

            logger.info("Extracted entities from text", count=len(entities))
            return entities

        except Exception as e:
            logger.error("Entity extraction failed", error=str(e))
            return []

    # ============================================================
    # JSON 结构化抽取 (规则)
    # ============================================================

    def extract_from_json(
        self,
        artifact: dict[str, Any],
        source_id: str = "",
    ) -> tuple[list[Entity], list[Relation]]:
        """
        从JSON结构化数据提取实体和关系（规则映射）

        Args:
            artifact: 文物JSON数据
            source_id: 来源文档ID（可选，默认使用 caseId）

        Returns:
            (实体列表, 关系列表)
        """
        case_id = artifact.get("caseId", "") or source_id
        entities = []

        # 1. 文物实体（核心）
        if entity := self._extract_artifact_entity(artifact, case_id):
            entities.append(entity)

        # 2. 年代实体
        if entity := self._extract_period_entity(artifact, case_id):
            entities.append(entity)

        # 3. 材质实体
        if entity := self._extract_material_entity(artifact, case_id):
            entities.append(entity)

        # 4. 行政区实体
        entities.extend(self._extract_location_entities(artifact, case_id))

        # 5. 资产类型实体
        entities.extend(self._extract_asset_type_entities(artifact, case_id))

        # 6. 保管单位实体
        entities.extend(self._extract_keeper_entities(artifact, case_id))

        # 7. 存放空间实体
        entities.extend(self._extract_space_entities(artifact, case_id))

        # 8. 人物实体（创作者等）
        entities.extend(self._extract_person_entities(artifact, case_id))

        # 9. 公告机关实体
        entities.extend(self._extract_gov_entities(artifact, case_id))

        # 10. 判定标准实体
        entities.extend(self._extract_criteria_entities(artifact, case_id))

        # 11. 来源方式实体
        if entity := self._extract_source_entity(artifact, case_id):
            entities.append(entity)

        # 关系抽取
        relations = self._extract_relations_from_json(artifact, entities)

        return entities, relations

    # ============================================================
    # Document 混合抽取
    # ============================================================

    async def extract_from_document(
        self,
        document: Document,
    ) -> tuple[list[Entity], list[Relation]]:
        """
        从Document对象提取实体和关系

        结合结构化字段提取和LLM补充提取

        Args:
            document: Document对象

        Returns:
            (实体列表, 关系列表)
        """
        # 1. 从结构化字段提取实体
        entities = self._extract_structured_entities(document)

        # 2. 从结构化字段提取关系
        relations = self._extract_structured_relations(document, entities)

        # 3. 从描述文本用LLM补充
        if self.enable_llm and document.description:
            text_entities, text_relations = await self.extract_from_text(
                document.description,
                source_doc_id=document.id,
            )
            entities.extend(text_entities)
            relations.extend(text_relations)

        return deduplicate_entities(entities), deduplicate_relations(relations)

    def _extract_structured_entities(self, document: Document) -> list[Entity]:
        """从文档结构化字段提取实体"""
        entities = []
        doc_id = document.id

        # 文物实体
        if document.artifact_name:
            name_normalized = (
                to_traditional(document.artifact_name)
                if self.config.normalize_to_traditional
                else document.artifact_name
            )
            entities.append(
                Entity(
                    id=generate_hash_id(document.artifact_id or doc_id),
                    name=document.artifact_name,
                    type=EntityType.ARTIFACT,
                    description=document.description or "",
                    attributes={
                        "artifact_id": document.artifact_id,
                        "dynasty": document.dynasty,
                        "period": document.period,
                        "material": document.material,
                        "location": document.location,
                        "dimensions": document.dimensions,
                    },
                    source_doc_id=doc_id,
                    name_normalized=name_normalized,
                )
            )

        # 朝代实体
        if document.dynasty:
            dynasty = extract_dynasty(document.dynasty) or document.dynasty
            entities.append(
                Entity(
                    id=generate_hash_id(f"{doc_id}_dynasty_{dynasty}"),
                    name=dynasty,
                    type=EntityType.DYNASTY,
                    description=f"朝代: {dynasty}",
                    source_doc_id=doc_id,
                )
            )

        # 年代/时期实体
        if document.period:
            year_ce, year_ce_end = normalize_year_to_ce(document.period)
            entities.append(
                Entity(
                    id=generate_hash_id(f"{doc_id}_period_{document.period}"),
                    name=document.period,
                    type=EntityType.PERIOD,
                    description=f"年代: {document.period}",
                    attributes={
                        "year_ce": year_ce,
                        "year_ce_end": year_ce_end,
                        "original": document.period,
                    },
                    source_doc_id=doc_id,
                )
            )

        # 材质实体
        if document.material:
            entities.append(
                Entity(
                    id=generate_hash_id(f"{doc_id}_material_{document.material}"),
                    name=document.material,
                    type=EntityType.MATERIAL,
                    description=f"材质: {document.material}",
                    source_doc_id=doc_id,
                )
            )

        # 地点实体
        if document.location:
            entities.append(
                Entity(
                    id=generate_hash_id(f"{doc_id}_location_{document.location}"),
                    name=document.location,
                    type=EntityType.LOCATION,
                    description=f"地点: {document.location}",
                    source_doc_id=doc_id,
                )
            )

        # 博物馆/收藏机构实体
        if document.museum:
            entities.append(
                Entity(
                    id=generate_hash_id(f"{doc_id}_museum_{document.museum}"),
                    name=document.museum,
                    type=EntityType.MUSEUM,
                    description=f"收藏机构: {document.museum}",
                    source_doc_id=doc_id,
                )
            )

        # 文化实体
        if document.culture:
            entities.append(
                Entity(
                    id=generate_hash_id(f"{doc_id}_culture_{document.culture}"),
                    name=document.culture,
                    type=EntityType.CULTURE,
                    description=f"文化: {document.culture}",
                    source_doc_id=doc_id,
                )
            )

        # 工艺实体
        if document.technique:
            entities.append(
                Entity(
                    id=generate_hash_id(f"{doc_id}_technique_{document.technique}"),
                    name=document.technique,
                    type=EntityType.TECHNIQUE,
                    description=f"工艺: {document.technique}",
                    source_doc_id=doc_id,
                )
            )

        return entities

    def _extract_structured_relations(
        self, document: Document, entities: list[Entity]
    ) -> list[Relation]:
        """从文档结构化字段提取关系"""
        relations = []
        doc_id = document.id

        # 构建实体名称到实体的映射
        entity_map = {e.name: e for e in entities}

        # 文物实体
        artifact_entity = entity_map.get(document.artifact_name) if document.artifact_name else None
        if not artifact_entity:
            return relations

        # 文物 - 朝代关系
        if document.dynasty:
            dynasty = extract_dynasty(document.dynasty) or document.dynasty
            if dynasty in entity_map:
                relations.append(
                    Relation.from_entities(
                        source=artifact_entity,
                        target=entity_map[dynasty],
                        relation_type=RelationType.BELONGS_TO_DYNASTY,
                        description=f"{document.artifact_name}属于{dynasty}",
                        source_doc_id=doc_id,
                    )
                )

        # 文物 - 年代关系
        if document.period and document.period in entity_map:
            relations.append(
                Relation.from_entities(
                    source=artifact_entity,
                    target=entity_map[document.period],
                    relation_type=RelationType.BELONGS_TO_PERIOD,
                    description=f"{document.artifact_name}的年代为{document.period}",
                    source_doc_id=doc_id,
                )
            )

        # 文物 - 材质关系
        if document.material and document.material in entity_map:
            relations.append(
                Relation.from_entities(
                    source=artifact_entity,
                    target=entity_map[document.material],
                    relation_type=RelationType.MADE_OF,
                    description=f"{document.artifact_name}的材质为{document.material}",
                    source_doc_id=doc_id,
                )
            )

        # 文物 - 地点关系
        if document.location and document.location in entity_map:
            relations.append(
                Relation.from_entities(
                    source=artifact_entity,
                    target=entity_map[document.location],
                    relation_type=RelationType.LOCATED_IN,
                    description=f"{document.artifact_name}位于{document.location}",
                    source_doc_id=doc_id,
                )
            )

        # 文物 - 博物馆关系
        if document.museum and document.museum in entity_map:
            relations.append(
                Relation.from_entities(
                    source=artifact_entity,
                    target=entity_map[document.museum],
                    relation_type=RelationType.KEPT_BY,
                    description=f"{document.artifact_name}收藏于{document.museum}",
                    source_doc_id=doc_id,
                )
            )

        # 文物 - 文化关系
        if document.culture and document.culture in entity_map:
            relations.append(
                Relation.from_entities(
                    source=artifact_entity,
                    target=entity_map[document.culture],
                    relation_type=RelationType.BELONGS_TO_CULTURE,
                    description=f"{document.artifact_name}属于{document.culture}文化",
                    source_doc_id=doc_id,
                )
            )

        # 文物 - 工艺关系
        if document.technique and document.technique in entity_map:
            relations.append(
                Relation.from_entities(
                    source=artifact_entity,
                    target=entity_map[document.technique],
                    relation_type=RelationType.USES_TECHNIQUE,
                    description=f"{document.artifact_name}采用{document.technique}工艺",
                    source_doc_id=doc_id,
                )
            )

        return relations

    # ============================================================
    # JSON 实体抽取私有方法
    # ============================================================

    def _extract_artifact_entity(self, artifact: dict[str, Any], case_id: str) -> Entity | None:
        """抽取文物实体"""
        case_name = artifact.get("caseName", "")
        if not case_name:
            return None

        name_normalized = (
            to_traditional(case_name) if self.config.normalize_to_traditional else case_name
        )

        return Entity(
            id=generate_hash_id(case_id),
            name=case_name,
            type=EntityType.ARTIFACT,
            description=artifact.get("reserveStatusDesc", "")[:200]
            if artifact.get("reserveStatusDesc")
            else "",
            attributes={
                "case_id": case_id,
                "assets_classify_name": artifact.get("assetsClassifyName", ""),
            },
            source_doc_id=case_id,
            name_normalized=name_normalized,
        )

    def _extract_period_entity(self, artifact: dict[str, Any], case_id: str) -> Entity | None:
        """抽取年代实体"""
        desc_age = artifact.get("descAge", "")
        if not desc_age:
            return None

        year_ce, year_ce_end = normalize_year_to_ce(desc_age)
        dynasty = extract_dynasty(desc_age)

        return Entity(
            id=generate_hash_id(f"{case_id}_age"),
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
        )

    def _extract_material_entity(self, artifact: dict[str, Any], case_id: str) -> Entity | None:
        """抽取材质实体"""
        material = artifact.get("descMaterial", "")
        if not material:
            return None

        return Entity(
            id=generate_hash_id(f"{case_id}_material"),
            name=material,
            type=EntityType.MATERIAL,
            description=f"材质: {material}",
            source_doc_id=case_id,
        )

    def _extract_location_entities(self, artifact: dict[str, Any], case_id: str) -> list[Entity]:
        """抽取行政区实体"""
        entities = []
        addresses = artifact.get("addresses", [])

        for addr in addresses:
            city = addr.get("cityName", "")
            district = addr.get("distName", "")

            if city:
                entities.append(
                    Entity(
                        id=generate_hash_id(f"{case_id}_city_{city}"),
                        name=city,
                        type=EntityType.LOCATION,
                        description=f"城市: {city}",
                        attributes={"level": "city"},
                        source_doc_id=case_id,
                    )
                )

            if city and district:
                full_district = f"{city}{district}"
                entities.append(
                    Entity(
                        id=generate_hash_id(f"{case_id}_district_{full_district}"),
                        name=full_district,
                        type=EntityType.LOCATION,
                        description=f"行政区: {full_district}",
                        attributes={"level": "district", "city": city, "district": district},
                        source_doc_id=case_id,
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
                    id=generate_hash_id(f"{case_id}_classify_{classify_name}"),
                    name=classify_name,
                    type=EntityType.ASSET_LEVEL,
                    description=f"资产分级: {classify_name}",
                    attributes={"code": artifact.get("assetsClassifyCode", "")},
                    source_doc_id=case_id,
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
                        id=generate_hash_id(f"{case_id}_type_{type_name}"),
                        name=type_name,
                        type=EntityType.ASSET_TYPE,
                        description=f"资产类型: {type_name}",
                        attributes={"code": atype.get("code", "")},
                        source_doc_id=case_id,
                    )
                )

            if sub_name:
                entities.append(
                    Entity(
                        id=generate_hash_id(f"{case_id}_subtype_{sub_name}"),
                        name=sub_name,
                        type=EntityType.ASSET_TYPE,
                        description=f"资产子类型: {sub_name}",
                        attributes={
                            "code": atype.get("subCode", ""),
                            "parent_type": type_name,
                        },
                        source_doc_id=case_id,
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
                        id=generate_hash_id(f"{case_id}_keeper_{name}"),
                        name=name,
                        type=EntityType.MUSEUM,
                        description=f"保管单位: {name}",
                        attributes={"role": "keeper"},
                        source_doc_id=case_id,
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
                        id=generate_hash_id(f"{case_id}_space_{save_space}"),
                        name=save_space,
                        type=EntityType.STORAGE_SPACE,
                        description=f"存放空间类型: {save_space}",
                        attributes={"identity": place.get("saveSpaceIdentity", "")},
                        source_doc_id=case_id,
                    )
                )

        return entities

    def _extract_person_entities(self, artifact: dict[str, Any], case_id: str) -> list[Entity]:
        """抽取人物实体"""
        entities = []

        creator = artifact.get("descCreator", "")
        if creator:
            entities.append(
                Entity(
                    id=generate_hash_id(f"{case_id}_creator_{creator}"),
                    name=creator,
                    type=EntityType.PERSON,
                    description=f"创作者: {creator}",
                    attributes={"role": "creator"},
                    source_doc_id=case_id,
                )
            )

        return entities

    def _extract_gov_entities(self, artifact: dict[str, Any], case_id: str) -> list[Entity]:
        """抽取政府机关实体"""
        entities = []

        gov_name = artifact.get("govInstitutionName", "")
        if gov_name:
            entities.append(
                Entity(
                    id=generate_hash_id(f"{case_id}_gov_{gov_name}"),
                    name=gov_name,
                    type=EntityType.GOV_INSTITUTION,
                    description=f"公告机关: {gov_name}",
                    source_doc_id=case_id,
                )
            )

        gov_inst = artifact.get("govInstitution", "")
        if gov_inst and gov_inst != gov_name:
            entities.append(
                Entity(
                    id=generate_hash_id(f"{case_id}_inst_{gov_inst}"),
                    name=gov_inst,
                    type=EntityType.ADMIN_AUTHORITY,
                    description=f"主管机关: {gov_inst}",
                    attributes={"dept": artifact.get("govDeptName", "")},
                    source_doc_id=case_id,
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
                        id=generate_hash_id(f"{case_id}_criteria_{criteria[:20]}"),
                        name=criteria,
                        type=EntityType.CRITERIA,
                        description=criteria,
                        source_doc_id=case_id,
                    )
                )

        return entities

    def _extract_source_entity(self, artifact: dict[str, Any], case_id: str) -> Entity | None:
        """抽取来源方式实体"""
        source = artifact.get("pastHistorySource", "")
        if not source:
            return None

        return Entity(
            id=generate_hash_id(f"{case_id}_source_{source}"),
            name=source,
            type=EntityType.PROVENANCE,
            description=f"来源方式: {source}",
            attributes={"notes": artifact.get("pastHistorySourceNotes", "")},
            source_doc_id=case_id,
        )

    # ============================================================
    # JSON 关系抽取私有方法
    # ============================================================

    def _extract_relations_from_json(
        self,
        artifact: dict[str, Any],
        entities: list[Entity],
    ) -> list[Relation]:
        """从JSON数据中抽取关系"""
        relations = []
        case_name = artifact.get("caseName", "")
        case_id = artifact.get("caseId", "")

        # 构建实体名称到实体的映射
        entity_map = {e.name: e for e in entities}

        # 文物实体
        artifact_entity = entity_map.get(case_name)
        if not artifact_entity:
            return relations

        # 1. 文物 - 年代关系
        desc_age = artifact.get("descAge", "")
        if desc_age and desc_age in entity_map:
            relations.append(
                Relation.from_entities(
                    source=artifact_entity,
                    target=entity_map[desc_age],
                    relation_type=RelationType.BELONGS_TO_PERIOD,
                    description=f"{case_name}的年代为{desc_age}",
                    source_doc_id=case_id,
                )
            )

        # 2. 文物 - 材质关系
        material = artifact.get("descMaterial", "")
        if material and material in entity_map:
            relations.append(
                Relation.from_entities(
                    source=artifact_entity,
                    target=entity_map[material],
                    relation_type=RelationType.MADE_OF,
                    description=f"{case_name}的材质为{material}",
                    source_doc_id=case_id,
                )
            )

        # 3. 文物 - 位置关系
        for addr in artifact.get("addresses", []):
            city = addr.get("cityName", "")
            district = addr.get("distName", "")
            if city and district:
                full_district = f"{city}{district}"
                if full_district in entity_map:
                    relations.append(
                        Relation.from_entities(
                            source=artifact_entity,
                            target=entity_map[full_district],
                            relation_type=RelationType.LOCATED_IN,
                            description=f"{case_name}位于{full_district}",
                            source_doc_id=case_id,
                        )
                    )

        # 4. 文物 - 资产等级关系
        classify_name = artifact.get("assetsClassifyName", "")
        if classify_name and classify_name in entity_map:
            relations.append(
                Relation.from_entities(
                    source=artifact_entity,
                    target=entity_map[classify_name],
                    relation_type=RelationType.BELONGS_TO_LEVEL,
                    description=f"{case_name}属于{classify_name}",
                    source_doc_id=case_id,
                )
            )

        # 5. 文物 - 资产类型关系
        for atype in artifact.get("assetsTypes", []):
            sub_name = atype.get("subName", "")
            if sub_name and sub_name in entity_map:
                relations.append(
                    Relation.from_entities(
                        source=artifact_entity,
                        target=entity_map[sub_name],
                        relation_type=RelationType.BELONGS_TO_TYPE,
                        description=f"{case_name}属于{sub_name}",
                        source_doc_id=case_id,
                    )
                )

        # 6. 文物 - 保管单位关系
        for dept in artifact.get("keepDepts", []):
            name = dept.get("name", "")
            if name and name in entity_map:
                relations.append(
                    Relation.from_entities(
                        source=artifact_entity,
                        target=entity_map[name],
                        relation_type=RelationType.KEPT_BY,
                        description=f"{case_name}由{name}保管",
                        source_doc_id=case_id,
                    )
                )

        # 7. 文物 - 存放空间关系
        for place in artifact.get("keepPlaces", []):
            save_space = place.get("saveSpace", "")
            if save_space and save_space in entity_map:
                relations.append(
                    Relation.from_entities(
                        source=artifact_entity,
                        target=entity_map[save_space],
                        relation_type=RelationType.STORED_IN,
                        description=f"{case_name}存放于{save_space}",
                        source_doc_id=case_id,
                    )
                )

        # 8. 文物 - 创作者关系
        creator = artifact.get("descCreator", "")
        if creator and creator in entity_map:
            relations.append(
                Relation.from_entities(
                    source=artifact_entity,
                    target=entity_map[creator],
                    relation_type=RelationType.CREATED_BY,
                    description=f"{case_name}的创作者是{creator}",
                    source_doc_id=case_id,
                )
            )

        # 9. 文物 - 公告机关关系
        gov_name = artifact.get("govInstitutionName", "")
        if gov_name and gov_name in entity_map:
            relations.append(
                Relation.from_entities(
                    source=artifact_entity,
                    target=entity_map[gov_name],
                    relation_type=RelationType.REGISTERED_BY,
                    description=f"{case_name}由{gov_name}登录",
                    source_doc_id=case_id,
                )
            )

        # 10. 文物 - 主管机关关系
        gov_inst = artifact.get("govInstitution", "")
        if gov_inst and gov_inst in entity_map and gov_inst != gov_name:
            relations.append(
                Relation.from_entities(
                    source=artifact_entity,
                    target=entity_map[gov_inst],
                    relation_type=RelationType.ADMINISTERED_BY,
                    description=f"{case_name}的主管机关是{gov_inst}",
                    source_doc_id=case_id,
                )
            )

        # 11. 文物 - 判定标准关系
        for criteria in artifact.get("judgeCriteria", []):
            if criteria and criteria in entity_map:
                relations.append(
                    Relation.from_entities(
                        source=artifact_entity,
                        target=entity_map[criteria],
                        relation_type=RelationType.MEETS_CRITERIA,
                        description=f"{case_name}符合标准: {criteria}",
                        source_doc_id=case_id,
                    )
                )

        # 12. 文物 - 来源方式关系
        source = artifact.get("pastHistorySource", "")
        if source and source in entity_map:
            relations.append(
                Relation.from_entities(
                    source=artifact_entity,
                    target=entity_map[source],
                    relation_type=RelationType.PROVENANCE_TYPE,
                    description=f"{case_name}的来源方式为{source}",
                    source_doc_id=case_id,
                )
            )

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
