"""
递归分块器

基于分隔符递归分割文本，适用于长文档
"""

from kg_search.ingestion.loaders.base import Document
from kg_search.utils import count_tokens

from .base import Chunk, TextChunker


class RecursiveChunker(TextChunker):
    """
    递归分块器

    使用分隔符优先级递归分割文本，适用于：
    - 长文本文档
    - Markdown文件
    - 纯文本文件

    对于结构化数据（JSON等），建议使用 StructureChunker
    """

    # 分隔符优先级（从高到低）
    SEPARATORS = [
        "\n\n\n",  # 多空行（章节分隔）
        "\n\n",  # 段落分隔
        "\n",  # 行分隔
        "。",  # 中文句号
        "！",  # 中文感叹号
        "？",  # 中文问号
        "；",  # 中文分号
        ". ",  # 英文句号
        "! ",  # 英文感叹号
        "? ",  # 英文问号
        "; ",  # 英文分号
        "，",  # 中文逗号（作为最后手段）
        ", ",  # 英文逗号
    ]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_tokens: int | None = None,
        keep_separator: bool = True,
        separators: list[str] | None = None,
    ):
        """
        初始化递归分块器

        Args:
            chunk_size: 目标块大小（字符数）
            chunk_overlap: 块重叠（字符数）
            max_tokens: 最大token数（如设置则以此为准）
            keep_separator: 是否在块中保留分隔符
            separators: 自定义分隔符列表（按优先级排序）
        """
        super().__init__(chunk_size, chunk_overlap)
        self.max_tokens = max_tokens
        self.keep_separator = keep_separator
        self.separators = separators or self.SEPARATORS

    def chunk(self, document: Document) -> list[Chunk]:
        """将文档分块"""
        text = document.content
        if not text or not text.strip():
            return []

        # 递归分割文本
        splits = self._split_text(text, self.separators)

        # 合并小块
        merged_chunks = self._merge_splits(splits)

        # 创建Chunk对象
        chunks = []
        char_offset = 0

        for i, chunk_text in enumerate(merged_chunks):
            # 计算位置
            start_char = text.find(chunk_text, char_offset)
            if start_char == -1:
                start_char = char_offset
            end_char = start_char + len(chunk_text)
            char_offset = start_char + 1  # 允许重叠

            chunk = Chunk(
                id=f"{document.id}_chunk_{i}",
                content=chunk_text,
                document_id=document.id,
                chunk_index=i,
                metadata=document.metadata,
                artifact_id=document.artifact_id,
                artifact_name=document.artifact_name,
                dynasty=document.dynasty,
                start_char=start_char,
                end_char=end_char,
            )
            chunks.append(chunk)

        return chunks

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """
        递归分割文本

        Args:
            text: 输入文本
            separators: 分隔符列表

        Returns:
            分割后的文本列表
        """
        if not separators:
            # 没有更多分隔符，按字符数强制分割
            return self._split_by_size(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        # 尝试使用当前分隔符分割
        if separator in text:
            splits = self._split_with_separator(text, separator)
        else:
            # 当前分隔符不存在，尝试下一个
            return self._split_text(text, remaining_separators)

        # 检查每个分割块是否需要进一步分割
        final_splits = []
        for split in splits:
            if self._is_chunk_too_large(split):
                # 递归分割
                final_splits.extend(self._split_text(split, remaining_separators))
            else:
                final_splits.append(split)

        return final_splits

    def _split_with_separator(self, text: str, separator: str) -> list[str]:
        """使用分隔符分割，可选保留分隔符"""
        if self.keep_separator:
            # 保留分隔符在块末尾
            parts = text.split(separator)
            splits = []
            for i, part in enumerate(parts):
                if i < len(parts) - 1:
                    splits.append(part + separator)
                elif part:  # 最后一个非空部分
                    splits.append(part)
            return [s for s in splits if s.strip()]
        else:
            return [s.strip() for s in text.split(separator) if s.strip()]

    def _split_by_size(self, text: str) -> list[str]:
        """按大小强制分割"""
        splits = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            splits.append(text[i : i + self.chunk_size])
        return splits

    def _is_chunk_too_large(self, text: str) -> bool:
        """检查块是否过大"""
        if self.max_tokens:
            return count_tokens(text) > self.max_tokens
        return len(text) > self.chunk_size

    def _merge_splits(self, splits: list[str]) -> list[str]:
        """
        合并过小的块

        Args:
            splits: 分割后的文本列表

        Returns:
            合并后的文本列表
        """
        if not splits:
            return []

        merged = []
        current_chunk = ""

        for split in splits:
            # 检查合并后是否超出限制
            test_chunk = current_chunk + split if current_chunk else split

            if self._is_chunk_too_large(test_chunk) and current_chunk:
                # 保存当前块，开始新块
                merged.append(current_chunk.strip())
                # 添加重叠部分
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + split
            else:
                current_chunk = test_chunk

        # 保存最后一个块
        if current_chunk.strip():
            merged.append(current_chunk.strip())

        return merged

    def _get_overlap_text(self, text: str) -> str:
        """获取重叠文本"""
        if len(text) <= self.chunk_overlap:
            return text
        return text[-self.chunk_overlap :]
