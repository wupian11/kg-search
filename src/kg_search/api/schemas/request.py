"""
请求模型定义
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class IngestTextRequest(BaseModel):
    """文本摄入请求"""

    content: str = Field(..., description="文本内容")
    doc_type: Literal["json", "jsonl", "md", "txt"] = Field(
        default="json",
        description="文档类型",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="附加元数据",
    )
    build_graph: bool = Field(
        default=True,
        description="是否构建知识图谱",
    )


class IngestFileRequest(BaseModel):
    """文件/目录摄入请求"""

    path: str = Field(..., description="文件或目录路径")
    recursive: bool = Field(
        default=True,
        description="是否递归处理子目录",
    )
    build_graph: bool = Field(
        default=True,
        description="是否构建知识图谱",
    )


class SearchRequest(BaseModel):
    """混合检索请求"""

    query: str = Field(..., description="查询文本", min_length=1)
    top_k: int = Field(default=10, description="返回结果数量", ge=1, le=100)
    filters: dict[str, Any] | None = Field(
        default=None,
        description="过滤条件 (向量检索)",
    )
    entity_types: list[str] | None = Field(
        default=None,
        description="实体类型过滤 (图检索)",
    )
    fusion_method: Literal["rrf", "weighted", "interleave"] = Field(
        default="rrf",
        description="融合方法",
    )


class VectorSearchRequest(BaseModel):
    """向量检索请求"""

    query: str = Field(..., description="查询文本", min_length=1)
    top_k: int = Field(default=10, description="返回结果数量", ge=1, le=100)
    filters: dict[str, Any] | None = Field(
        default=None,
        description="过滤条件",
    )
    min_score: float = Field(
        default=0.0,
        description="最小相似度分数",
        ge=0.0,
        le=1.0,
    )


class GraphSearchRequest(BaseModel):
    """图检索请求"""

    query: str = Field(default="", description="查询文本")
    entity_name: str | None = Field(
        default=None,
        description="实体名称 (用于实体中心检索)",
    )
    entity_types: list[str] | None = Field(
        default=None,
        description="实体类型过滤",
    )
    relation_types: list[str] | None = Field(
        default=None,
        description="关系类型过滤",
    )
    depth: int = Field(
        default=2,
        description="图遍历深度",
        ge=1,
        le=5,
    )
    top_k: int = Field(default=10, description="返回结果数量", ge=1, le=100)


class QuestionRequest(BaseModel):
    """问答请求"""

    question: str = Field(..., description="用户问题", min_length=1)
    top_k: int = Field(default=10, description="检索结果数量", ge=1, le=50)
    generate_answer: bool = Field(
        default=True,
        description="是否生成答案",
    )
    stream: bool = Field(
        default=False,
        description="是否流式输出",
    )
    search_type: Literal["auto", "local", "global", "hybrid"] = Field(
        default="auto",
        description="检索类型",
    )
