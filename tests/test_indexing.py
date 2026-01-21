"""
索引存储层测试
"""

import pytest
from pathlib import Path

from kg_search.indexing import DocumentStore
from kg_search.utils.models import Document, Chunk


class TestDocumentStore:
    """文档存储测试"""

    @pytest.mark.asyncio
    async def test_add_and_get_document(self, temp_dir: Path, sample_document: Document):
        """测试添加和获取文档"""
        store = DocumentStore(str(temp_dir / "docs"))

        # 添加文档
        await store.add_documents([sample_document])

        # 获取文档
        doc = await store.get_document(sample_document.id)

        assert doc is not None
        assert doc.id == sample_document.id
        assert doc.content == sample_document.content

    @pytest.mark.asyncio
    async def test_add_and_get_chunks(self, temp_dir: Path, sample_chunks: list[Chunk]):
        """测试添加和获取文本块"""
        store = DocumentStore(str(temp_dir / "docs"))

        # 添加文本块
        await store.add_chunks(sample_chunks)

        # 获取文本块
        chunk = await store.get_chunk(sample_chunks[0].id)

        assert chunk is not None
        assert chunk.id == sample_chunks[0].id

    @pytest.mark.asyncio
    async def test_list_documents(self, temp_dir: Path, sample_document: Document):
        """测试列出文档"""
        store = DocumentStore(str(temp_dir / "docs"))

        await store.add_documents([sample_document])

        docs = await store.list_documents()

        assert len(docs) == 1
        assert docs[0].id == sample_document.id
