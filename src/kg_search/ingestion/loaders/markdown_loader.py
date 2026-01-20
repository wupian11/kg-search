"""
Markdown文档加载器
"""

import re
from pathlib import Path

from kg_search.utils import generate_id

from .base import Document, DocumentLoader


class MarkdownLoader(DocumentLoader):
    """Markdown文档加载器"""

    supported_extensions = [".md", ".markdown"]

    def __init__(self, extract_metadata: bool = True):
        """
        初始化Markdown加载器

        Args:
            extract_metadata: 是否尝试从frontmatter提取元数据
        """
        self.extract_metadata = extract_metadata

    def load(self, file_path: str | Path) -> list[Document]:
        """加载Markdown文件"""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        return self.load_from_string(content, source=str(path))

    def load_from_string(self, content: str, source: str = "") -> list[Document]:
        """从字符串加载Markdown"""
        metadata = {}
        doc_content = content

        # 提取YAML frontmatter
        if self.extract_metadata:
            metadata, doc_content = self._extract_frontmatter(content)

        # 尝试从内容中提取文物信息
        artifact_info = self._extract_artifact_info(doc_content)

        doc_id = metadata.get("id") or artifact_info.get("artifact_id") or generate_id("md")

        return [
            Document(
                id=doc_id,
                content=doc_content,
                metadata=metadata,
                source=source,
                doc_type="markdown",
                artifact_id=artifact_info.get("artifact_id"),
                artifact_name=artifact_info.get("artifact_name"),
                dynasty=artifact_info.get("dynasty"),
                material=artifact_info.get("material"),
                location=artifact_info.get("location"),
                museum=artifact_info.get("museum"),
                description=artifact_info.get("description"),
            )
        ]

    def _extract_frontmatter(self, content: str) -> tuple[dict, str]:
        """
        提取YAML frontmatter

        Args:
            content: Markdown内容

        Returns:
            (元数据字典, 去除frontmatter后的内容)
        """
        frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.match(frontmatter_pattern, content, re.DOTALL)

        if not match:
            return {}, content

        frontmatter_text = match.group(1)
        doc_content = content[match.end() :]

        # 简单解析YAML（避免引入yaml依赖）
        metadata = {}
        for line in frontmatter_text.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip().strip("\"'")

        return metadata, doc_content

    def _extract_artifact_info(self, content: str) -> dict:
        """
        从Markdown内容中提取文物信息

        支持的格式：
        - 标题作为文物名称
        - 表格或列表形式的属性

        Args:
            content: Markdown内容

        Returns:
            提取的文物信息
        """
        info = {}

        # 提取第一个一级标题作为文物名称
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if title_match:
            info["artifact_name"] = title_match.group(1).strip()

        # 提取常见字段（支持"字段：值"或"- 字段：值"格式）
        field_patterns = {
            "dynasty": r"(?:朝代|时代)[：:]\s*(.+)",
            "material": r"材质[：:]\s*(.+)",
            "location": r"(?:出土地点|发现地)[：:]\s*(.+)",
            "museum": r"(?:收藏|馆藏)[：:]\s*(.+)",
        }

        for field, pattern in field_patterns.items():
            match = re.search(pattern, content)
            if match:
                info[field] = match.group(1).strip()

        return info
