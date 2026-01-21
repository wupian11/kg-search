"""
测试配置和fixtures
"""

import asyncio
import pytest
from typing import Generator, AsyncGenerator
from pathlib import Path
import tempfile
import shutil

from kg_search.config import get_settings, Settings
from kg_search.ingestion.loaders.base import Document
from kg_search.ingestion.chunkers.base import Chunk
from kg_search.extraction.entity_extractor import Entity
from kg_search.extraction.relation_extractor import Relation
from kg_search.extraction.graph_builder import KnowledgeGraph


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def settings() -> Settings:
    """获取测试配置"""
    return get_settings()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """创建临时目录"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_document() -> Document:
    """示例文档"""
    return Document(
        id="doc_001",
        content="四羊方尊是商代晚期青铜礼器，属于祭祀用的酒器。方尊四角各有一只卷角羊。",
        metadata={
            "name": "四羊方尊",
            "dynasty": "商代",
            "material": "青铜",
            "museum": "中国国家博物馆",
        },
        source="test.json",
    )


@pytest.fixture
def sample_chunks(sample_document: Document) -> list[Chunk]:
    """示例文本块"""
    return [
        Chunk(
            id="chunk_001",
            document_id=sample_document.id,
            content="四羊方尊是商代晚期青铜礼器，属于祭祀用的酒器。",
            metadata=sample_document.metadata,
            chunk_index=0,
        ),
        Chunk(
            id="chunk_002",
            document_id=sample_document.id,
            content="方尊四角各有一只卷角羊，造型生动。",
            metadata=sample_document.metadata,
            chunk_index=1,
        ),
    ]


@pytest.fixture
def sample_entities() -> list[Entity]:
    """示例实体"""
    return [
        Entity(
            id="entity_001",
            name="四羊方尊",
            type="文物",
            description="商代晚期青铜礼器",
            properties={"category": "酒器"},
        ),
        Entity(
            id="entity_002", name="商代", type="朝代", description="中国历史朝代", properties={}
        ),
        Entity(id="entity_003", name="青铜", type="材质", description="铜合金材料", properties={}),
        Entity(
            id="entity_004",
            name="中国国家博物馆",
            type="收藏机构",
            description="位于北京的国家级博物馆",
            properties={},
        ),
    ]


@pytest.fixture
def sample_relations(sample_entities: list[Entity]) -> list[Relation]:
    """示例关系"""
    return [
        Relation(
            id="rel_001",
            source=sample_entities[0].id,
            target=sample_entities[1].id,
            type="属于朝代",
            properties={},
        ),
        Relation(
            id="rel_002",
            source=sample_entities[0].id,
            target=sample_entities[2].id,
            type="材质为",
            properties={},
        ),
        Relation(
            id="rel_003",
            source=sample_entities[0].id,
            target=sample_entities[3].id,
            type="收藏于",
            properties={},
        ),
    ]


@pytest.fixture
def sample_knowledge_graph(
    sample_entities: list[Entity], sample_relations: list[Relation]
) -> KnowledgeGraph:
    """示例知识图谱"""
    return KnowledgeGraph(entities=sample_entities, relations=sample_relations)


@pytest.fixture
def sample_json_data() -> dict:
    """示例JSON数据（嵌套格式）"""
    return {
        "artifacts": [
            {
                "id": "artifact_001",
                "basic_info": {
                    "name": "四羊方尊",
                    "dynasty": "商代",
                    "period": "公元前14世纪-前11世纪",
                },
                "physical_info": {"material": "青铜", "technique": "分铸法"},
                "provenance": {"excavation_site": "湖南宁乡", "museum": "中国国家博物馆"},
                "description": "四羊方尊是商代晚期青铜礼器。",
            }
        ]
    }


@pytest.fixture
def sample_markdown_content() -> str:
    """示例Markdown内容"""
    return """# 四羊方尊

## 基本信息
- 朝代：商代
- 年代：公元前14世纪-前11世纪

## 物理特征
- 材质：青铜
- 工艺：分铸法

## 描述
四羊方尊是商代晚期青铜礼器，属于祭祀用的酒器。
"""
