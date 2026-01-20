"""
数据摄入管道

协调文档加载、分块和索引构建的完整流程
"""

from pathlib import Path
from typing import Any

from kg_search.config import get_settings
from kg_search.utils import get_logger

from .chunkers import Chunk, SemanticChunker
from .loaders import Document, DocumentLoader, JSONLLoader, JSONLoader, MarkdownLoader, TextLoader

logger = get_logger(__name__)


class IngestionPipeline:
    """数据摄入管道"""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        max_tokens: int | None = None,
    ):
        """
        初始化摄入管道

        Args:
            chunk_size: 块大小
            chunk_overlap: 块重叠
            max_tokens: 最大token数
        """
        settings = get_settings()

        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.max_tokens = max_tokens or settings.max_tokens_per_chunk

        # 初始化加载器
        self.loaders: dict[str, DocumentLoader] = {
            ".json": JSONLoader(),
            ".jsonl": JSONLLoader(),
            ".md": MarkdownLoader(),
            ".markdown": MarkdownLoader(),
            ".txt": TextLoader(),
        }

        # 初始化分块器
        self.chunker = SemanticChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            max_tokens=self.max_tokens,
        )

    def process_file(self, file_path: str | Path) -> tuple[list[Document], list[Chunk]]:
        """
        处理单个文件

        Args:
            file_path: 文件路径

        Returns:
            (文档列表, 块列表)
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # 获取对应的加载器
        loader = self.loaders.get(path.suffix.lower())
        if not loader:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        logger.info("Processing file", file=str(path))

        # 加载文档
        documents = loader.load(path)
        logger.info("Loaded documents", count=len(documents))

        # 分块
        chunks = self.chunker.chunk_documents(documents)
        logger.info("Created chunks", count=len(chunks))

        return documents, chunks

    def process_directory(
        self,
        dir_path: str | Path,
        recursive: bool = True,
    ) -> tuple[list[Document], list[Chunk]]:
        """
        处理目录中的所有文件

        Args:
            dir_path: 目录路径
            recursive: 是否递归处理子目录

        Returns:
            (所有文档列表, 所有块列表)
        """
        path = Path(dir_path)

        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        all_documents: list[Document] = []
        all_chunks: list[Chunk] = []

        # 收集所有支持的文件
        pattern = "**/*" if recursive else "*"
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.loaders:
                try:
                    documents, chunks = self.process_file(file_path)
                    all_documents.extend(documents)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error("Failed to process file", file=str(file_path), error=str(e))

        logger.info(
            "Directory processing complete",
            total_documents=len(all_documents),
            total_chunks=len(all_chunks),
        )

        return all_documents, all_chunks

    def process_text(
        self,
        text: str,
        doc_type: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[Document], list[Chunk]]:
        """
        处理文本字符串

        Args:
            text: 输入文本
            doc_type: 文档类型 (json, md, txt)
            metadata: 附加元数据

        Returns:
            (文档列表, 块列表)
        """
        # 选择加载器
        loader_map = {
            "json": JSONLoader(),
            "jsonl": JSONLLoader(),
            "md": MarkdownLoader(),
            "markdown": MarkdownLoader(),
            "txt": TextLoader(),
            "text": TextLoader(),
        }

        loader = loader_map.get(doc_type.lower())
        if not loader:
            raise ValueError(f"Unsupported doc_type: {doc_type}")

        # 加载
        documents = loader.load_from_string(text, source="inline")

        # 附加元数据
        if metadata:
            for doc in documents:
                doc.metadata.update(metadata)

        # 分块
        chunks = self.chunker.chunk_documents(documents)

        return documents, chunks
