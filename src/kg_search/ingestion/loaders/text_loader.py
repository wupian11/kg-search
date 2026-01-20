"""
纯文本文档加载器
"""

from pathlib import Path

from kg_search.utils import generate_id

from .base import Document, DocumentLoader


class TextLoader(DocumentLoader):
    """纯文本文档加载器"""

    supported_extensions = [".txt"]

    def load(self, file_path: str | Path) -> list[Document]:
        """加载文本文件"""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        return self.load_from_string(content, source=str(path))

    def load_from_string(self, content: str, source: str = "") -> list[Document]:
        """从字符串加载文本"""
        doc_id = generate_id("txt")

        return [
            Document(
                id=doc_id,
                content=content,
                metadata={},
                source=source,
                doc_type="text",
            )
        ]
