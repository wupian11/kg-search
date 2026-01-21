"""
知识提取层测试
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from kg_search.extraction import EntityExtractor, RelationExtractor, GraphBuilder
from kg_search.utils.models import Entity, Relation


class TestEntityExtractor:
    """实体提取器测试"""

    @pytest.mark.asyncio
    async def test_extract_entities(self):
        """测试实体提取"""
        # Mock LLM client
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(
            return_value="""
[
    {
        "name": "四羊方尊",
        "type": "文物",
        "description": "商代晚期青铜礼器"
    },
    {
        "name": "商代",
        "type": "朝代",
        "description": "中国历史朝代"
    }
]
"""
        )

        extractor = EntityExtractor(mock_client)
        text = "四羊方尊是商代晚期青铜礼器"

        entities = await extractor.extract(text)

        assert len(entities) >= 2
        assert any(e.name == "四羊方尊" for e in entities)


class TestRelationExtractor:
    """关系提取器测试"""

    @pytest.mark.asyncio
    async def test_extract_relations(self, sample_entities):
        """测试关系提取"""
        # Mock LLM client
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(
            return_value="""
[
    {
        "source": "四羊方尊",
        "target": "商代",
        "type": "属于朝代"
    }
]
"""
        )

        extractor = RelationExtractor(mock_client)
        text = "四羊方尊是商代晚期青铜礼器"

        relations = await extractor.extract(text, sample_entities)

        assert len(relations) >= 1


class TestGraphBuilder:
    """图构建器测试"""

    @pytest.mark.asyncio
    async def test_build_from_text(self):
        """测试从文本构建知识图谱"""
        # Mock LLM client
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(
            side_effect=[
                # 实体提取响应
                """[{"name": "四羊方尊", "type": "文物", "description": "商代青铜器"}]""",
                # 关系提取响应
                """[]""",
            ]
        )

        builder = GraphBuilder(mock_client)
        text = "四羊方尊是商代晚期青铜礼器"

        kg = await builder.build_from_text(text)

        assert len(kg.entities) >= 1
