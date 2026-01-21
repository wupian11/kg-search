"""
GraphRAG配置

GraphRAG相关的配置管理
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from kg_search.config import get_settings


@dataclass
class GraphRAGConfig:
    """GraphRAG配置类"""

    # 实体抽取配置
    entity_extract_max_tokens: int = 4000
    entity_extract_prompt_file: str | None = None
    encoding_model: str = "cl100k_base"

    # 关系抽取配置
    max_relationships_per_chunk: int = 20
    relationship_extract_prompt_file: str | None = None

    # 社区检测配置
    community_algorithm: Literal["leiden", "louvain"] = "leiden"
    max_community_size: int = 100
    community_levels: list[int] = field(default_factory=lambda: [0, 1, 2])

    # 摘要生成配置
    summarize_max_tokens: int = 500
    summarize_prompt_file: str | None = None

    # Global Search配置
    global_search_max_tokens: int = 12000
    map_max_tokens: int = 1000
    reduce_max_tokens: int = 2000
    concurrency: int = 32

    # Local Search配置
    local_search_max_tokens: int = 12000
    local_search_text_unit_count: int = 20
    local_search_community_level: int = 2

    # LLM配置
    llm_model: str = ""
    embedding_model: str = ""
    api_key: str = ""
    api_base: str = ""

    @classmethod
    def from_settings(cls) -> "GraphRAGConfig":
        """从应用设置创建配置"""
        settings = get_settings()

        return cls(
            community_algorithm=settings.community_detection_algorithm,
            max_community_size=settings.community_max_size,
            summarize_max_tokens=settings.community_summary_max_tokens,
            llm_model=settings.current_llm_model,
            embedding_model=settings.current_embedding_model,
            api_key=settings.current_llm_api_key,
            api_base=settings.current_llm_api_base,
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "entity_extract_max_tokens": self.entity_extract_max_tokens,
            "max_relationships_per_chunk": self.max_relationships_per_chunk,
            "community_algorithm": self.community_algorithm,
            "max_community_size": self.max_community_size,
            "community_levels": self.community_levels,
            "summarize_max_tokens": self.summarize_max_tokens,
            "global_search_max_tokens": self.global_search_max_tokens,
            "map_max_tokens": self.map_max_tokens,
            "reduce_max_tokens": self.reduce_max_tokens,
            "local_search_max_tokens": self.local_search_max_tokens,
            "local_search_text_unit_count": self.local_search_text_unit_count,
        }


# 文博领域自定义实体类型
CULTURAL_RELIC_ENTITY_TYPES = [
    "文物",  # Artifact
    "朝代",  # Dynasty
    "年代",  # Period
    "材质",  # Material
    "工艺",  # Technique
    "地点",  # Location
    "收藏机构",  # Museum
    "人物",  # Person
    "文化",  # Culture
    "尺寸",  # Dimension
    "风格",  # Style
    "事件",  # Event
]

# 文博领域自定义关系类型
CULTURAL_RELIC_RELATION_TYPES = [
    "属于朝代",  # BELONGS_TO_DYNASTY
    "出土于",  # EXCAVATED_FROM
    "收藏于",  # COLLECTED_BY
    "材质为",  # MADE_OF
    "制作者",  # CREATED_BY
    "发现者",  # DISCOVERED_BY
    "同时期",  # SAME_PERIOD
    "风格相似",  # SIMILAR_STYLE
    "工艺传承",  # TECHNIQUE_INHERIT
    "属于文化",  # BELONGS_TO_CULTURE
    "位于",  # LOCATED_IN
    "相关",  # RELATED_TO
]
