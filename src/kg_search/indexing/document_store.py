"""
文档存储

存储原始文档和元数据
"""

import json
from pathlib import Path
from typing import Any

from kg_search.config import get_settings
from kg_search.ingestion.chunkers.base import Chunk
from kg_search.ingestion.loaders.base import Document
from kg_search.utils import get_logger

logger = get_logger(__name__)


class DocumentStore:
    """文档存储（基于文件系统）"""

    def __init__(self, base_dir: str | None = None):
        """
        初始化文档存储

        Args:
            base_dir: 存储基础目录
        """
        settings = get_settings()
        self.base_dir = Path(base_dir or settings.processed_data_dir)

        # 创建目录结构
        self.documents_dir = self.base_dir / "documents"
        self.chunks_dir = self.base_dir / "chunks"
        self.metadata_dir = self.base_dir / "metadata"

        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # 索引文件
        self.index_file = self.metadata_dir / "index.json"
        self._load_index()

        logger.info("DocumentStore initialized", base_dir=str(self.base_dir))

    def _load_index(self) -> None:
        """加载索引"""
        if self.index_file.exists():
            with open(self.index_file, "r", encoding="utf-8") as f:
                self.index = json.load(f)
        else:
            self.index = {
                "documents": {},
                "chunks": {},
            }

    def _save_index(self) -> None:
        """保存索引"""
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)

    async def add_document(self, document: Document) -> None:
        """
        添加文档

        Args:
            document: 文档对象
        """
        # 保存文档
        doc_file = self.documents_dir / f"{document.id}.json"
        with open(doc_file, "w", encoding="utf-8") as f:
            json.dump(document.to_dict(), f, ensure_ascii=False, indent=2)

        # 更新索引
        self.index["documents"][document.id] = {
            "file": str(doc_file),
            "artifact_name": document.artifact_name,
            "dynasty": document.dynasty,
            "source": document.source,
        }
        self._save_index()

        logger.debug("Document saved", doc_id=document.id)

    async def add_documents(self, documents: list[Document]) -> None:
        """批量添加文档"""
        for doc in documents:
            await self.add_document(doc)
        logger.info("Documents saved", count=len(documents))

    async def add_chunk(self, chunk: Chunk) -> None:
        """
        添加文本块

        Args:
            chunk: 文本块对象
        """
        chunk_file = self.chunks_dir / f"{chunk.id}.json"
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(chunk.to_dict(), f, ensure_ascii=False, indent=2)

        # 更新索引
        self.index["chunks"][chunk.id] = {
            "file": str(chunk_file),
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
        }
        self._save_index()

    async def add_chunks(self, chunks: list[Chunk]) -> None:
        """批量添加文本块"""
        for chunk in chunks:
            await self.add_chunk(chunk)
        logger.info("Chunks saved", count=len(chunks))

    async def get_document(self, doc_id: str) -> Document | None:
        """
        获取文档

        Args:
            doc_id: 文档ID

        Returns:
            文档对象或None
        """
        if doc_id not in self.index["documents"]:
            return None

        doc_file = Path(self.index["documents"][doc_id]["file"])
        if not doc_file.exists():
            return None

        with open(doc_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return Document.from_dict(data)

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        """
        获取文本块

        Args:
            chunk_id: 块ID

        Returns:
            文本块对象或None
        """
        if chunk_id not in self.index["chunks"]:
            return None

        chunk_file = Path(self.index["chunks"][chunk_id]["file"])
        if not chunk_file.exists():
            return None

        with open(chunk_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return Chunk(**data)

    async def get_chunks_by_document(self, doc_id: str) -> list[Chunk]:
        """
        获取文档的所有块

        Args:
            doc_id: 文档ID

        Returns:
            文本块列表
        """
        chunks = []
        for chunk_id, chunk_info in self.index["chunks"].items():
            if chunk_info["document_id"] == doc_id:
                chunk = await self.get_chunk(chunk_id)
                if chunk:
                    chunks.append(chunk)

        # 按索引排序
        chunks.sort(key=lambda c: c.chunk_index)
        return chunks

    async def search_documents(
        self,
        artifact_name: str | None = None,
        dynasty: str | None = None,
    ) -> list[Document]:
        """
        搜索文档

        Args:
            artifact_name: 文物名称（模糊匹配）
            dynasty: 朝代

        Returns:
            文档列表
        """
        results = []

        for doc_id, doc_info in self.index["documents"].items():
            match = True

            if artifact_name:
                if (
                    not doc_info.get("artifact_name")
                    or artifact_name not in doc_info["artifact_name"]
                ):
                    match = False

            if dynasty:
                if doc_info.get("dynasty") != dynasty:
                    match = False

            if match:
                doc = await self.get_document(doc_id)
                if doc:
                    results.append(doc)

        return results

    async def delete_document(self, doc_id: str) -> bool:
        """
        删除文档及其所有块

        Args:
            doc_id: 文档ID

        Returns:
            是否成功
        """
        if doc_id not in self.index["documents"]:
            return False

        # 删除文档文件
        doc_file = Path(self.index["documents"][doc_id]["file"])
        if doc_file.exists():
            doc_file.unlink()

        # 删除相关的块
        chunks_to_delete = [
            cid for cid, cinfo in self.index["chunks"].items() if cinfo["document_id"] == doc_id
        ]

        for chunk_id in chunks_to_delete:
            chunk_file = Path(self.index["chunks"][chunk_id]["file"])
            if chunk_file.exists():
                chunk_file.unlink()
            del self.index["chunks"][chunk_id]

        # 更新索引
        del self.index["documents"][doc_id]
        self._save_index()

        logger.info("Document deleted", doc_id=doc_id, chunks_deleted=len(chunks_to_delete))
        return True

    def get_stats(self) -> dict[str, Any]:
        """获取存储统计信息"""
        return {
            "total_documents": len(self.index["documents"]),
            "total_chunks": len(self.index["chunks"]),
            "storage_dir": str(self.base_dir),
        }
