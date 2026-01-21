"""
数据摄入层测试
"""

import json
import pytest
from pathlib import Path

from kg_search.ingestion.loaders import JSONLoader, JSONLLoader, MarkdownLoader, TextLoader
from kg_search.ingestion.chunkers import SemanticChunker
from kg_search.ingestion import IngestionPipeline


class TestJSONLoader:
    """JSON加载器测试"""

    def test_load_nested_json(self, temp_dir: Path, sample_json_data: dict):
        """测试嵌套JSON加载"""
        # 创建测试文件
        json_file = temp_dir / "test.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_json_data, f, ensure_ascii=False)

        # 加载
        loader = JSONLoader(
            content_fields=["description"],
            metadata_fields=["basic_info.name", "basic_info.dynasty", "physical_info.material"],
            nested_path="artifacts",
        )

        documents = loader.load(str(json_file))

        assert len(documents) == 1
        assert "四羊方尊是商代晚期青铜礼器" in documents[0].content
        assert documents[0].metadata.get("basic_info.name") == "四羊方尊"


class TestMarkdownLoader:
    """Markdown加载器测试"""

    def test_load_markdown(self, temp_dir: Path, sample_markdown_content: str):
        """测试Markdown加载"""
        md_file = temp_dir / "test.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(sample_markdown_content)

        loader = MarkdownLoader(extract_metadata=True)
        documents = loader.load(str(md_file))

        assert len(documents) == 1
        assert "四羊方尊" in documents[0].content


class TestSemanticChunker:
    """语义分块器测试"""

    def test_chunk_document(self, sample_document):
        """测试文档分块"""
        chunker = SemanticChunker(chunk_size=50, chunk_overlap=10)

        chunks = chunker.chunk(sample_document)

        assert len(chunks) > 0
        assert all(chunk.document_id == sample_document.id for chunk in chunks)
        assert all(len(chunk.content) <= 100 for chunk in chunks)  # 允许一定的溢出


class TestIngestionPipeline:
    """摄入管道测试"""

    def test_process_text(self):
        """测试文本处理"""
        pipeline = IngestionPipeline()

        text = "这是一段关于文物的测试文本。"
        documents, chunks = pipeline.process_text(text, source="test")

        assert len(documents) == 1
        assert len(chunks) > 0
