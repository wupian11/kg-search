"""
响应模型定义
"""

from typing import Any

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    """摄入响应"""

    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="消息")
    documents_count: int = Field(default=0, description="文档数量")
    chunks_count: int = Field(default=0, description="文本块数量")
    graph_stats: dict[str, int] | None = Field(
        default=None,
        description="图谱统计 (实体数、关系数)",
    )


class TaskResponse(BaseModel):
    """异步任务响应"""

    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    message: str = Field(default="", description="消息")


class SearchResult(BaseModel):
    """单条检索结果"""

    id: str = Field(..., description="结果ID")
    content: str = Field(..., description="内容")
    score: float = Field(..., description="相关性分数")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")
    source_type: str = Field(default="unknown", description="来源类型")


class SearchResponse(BaseModel):
    """检索响应"""

    query: str = Field(..., description="查询文本")
    results: list[SearchResult] = Field(default_factory=list, description="检索结果")
    total: int = Field(default=0, description="结果总数")


class QuestionResponse(BaseModel):
    """问答响应"""

    question: str = Field(..., description="用户问题")
    answer: str | None = Field(default=None, description="生成的答案")
    context: str = Field(default="", description="检索到的上下文")
    sources: list[SearchResult] = Field(default_factory=list, description="来源引用")


class EntityResponse(BaseModel):
    """实体响应"""

    id: str
    name: str
    type: str
    description: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)


class RelationResponse(BaseModel):
    """关系响应"""

    id: str
    source_entity: str
    target_entity: str
    relation_type: str
    description: str = ""


class GraphResponse(BaseModel):
    """知识图谱响应"""

    entities: list[EntityResponse] = Field(default_factory=list)
    relations: list[RelationResponse] = Field(default_factory=list)
    total_entities: int = 0
    total_relations: int = 0
