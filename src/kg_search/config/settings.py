"""
应用配置管理

使用 Pydantic Settings 管理所有配置项
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置类"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ============ 应用配置 ============
    app_name: str = "KG-Search"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ============ API配置 ============
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = Field(default="", description="API访问密钥")
    api_key_header: str = "X-API-Key"

    # ============ LLM服务提供商选择 ============
    llm_provider: Literal["openai", "glm"] = Field(
        default="openai", description="LLM服务提供商: openai 或 glm"
    )
    embedding_provider: Literal["openai", "glm"] = Field(
        default="openai", description="Embedding服务提供商: openai 或 glm"
    )

    # ============ OpenAI配置 ============
    openai_api_key: str = Field(default="", description="OpenAI API密钥")
    openai_api_base: str = Field(
        default="https://api.openai.com/v1", description="OpenAI API基础URL"
    )
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1536
    openai_max_retries: int = 3
    openai_timeout: int = 60

    # ============ GLM(智谱AI)配置 ============
    glm_api_key: str = Field(default="", description="智谱AI API密钥")
    glm_api_base: str = Field(
        default="https://open.bigmodel.cn/api/paas/v4", description="智谱AI API基础URL"
    )
    glm_model: str = "glm-4-flash"
    glm_embedding_model: str = "embedding-3"
    glm_embedding_dimensions: int = 2048
    glm_max_retries: int = 3
    glm_timeout: int = 60

    # ============ ChromaDB配置 ============
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection_name: str = "artifacts"
    chroma_persist_directory: str = "./data/chroma"

    # ============ Neo4j配置 ============
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = Field(default="", description="Neo4j密码")
    neo4j_database: str = "neo4j"

    # ============ 文档处理配置 ============
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_tokens_per_chunk: int = 500

    # ============ 检索配置 ============
    vector_search_top_k: int = 10
    graph_search_depth: int = 2
    hybrid_search_alpha: float = 0.5  # 向量检索权重，1-alpha为图检索权重

    # ============ GraphRAG配置 ============
    community_detection_algorithm: Literal["leiden", "louvain"] = "leiden"
    community_max_size: int = 100
    community_summary_max_tokens: int = 500

    # ============ 数据目录 ============
    data_dir: str = "./data"
    raw_data_dir: str = "./data/raw"
    processed_data_dir: str = "./data/processed"

    # ============ 便捷属性 ============
    @property
    def current_llm_api_key(self) -> str:
        """获取当前LLM提供商的API密钥"""
        return self.glm_api_key if self.llm_provider == "glm" else self.openai_api_key

    @property
    def current_llm_api_base(self) -> str:
        """获取当前LLM提供商的API基础URL"""
        return self.glm_api_base if self.llm_provider == "glm" else self.openai_api_base

    @property
    def current_llm_model(self) -> str:
        """获取当前LLM模型名称"""
        return self.glm_model if self.llm_provider == "glm" else self.openai_model

    @property
    def current_embedding_api_key(self) -> str:
        """获取当前Embedding提供商的API密钥"""
        return self.glm_api_key if self.embedding_provider == "glm" else self.openai_api_key

    @property
    def current_embedding_api_base(self) -> str:
        """获取当前Embedding提供商的API基础URL"""
        return self.glm_api_base if self.embedding_provider == "glm" else self.openai_api_base

    @property
    def current_embedding_model(self) -> str:
        """获取当前Embedding模型名称"""
        return self.glm_embedding_model if self.embedding_provider == "glm" else self.openai_embedding_model

    @property
    def current_embedding_dimensions(self) -> int:
        """获取当前Embedding向量维度"""
        return self.glm_embedding_dimensions if self.embedding_provider == "glm" else self.openai_embedding_dimensions


@lru_cache
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()
