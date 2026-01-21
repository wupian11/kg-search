"""
依赖注入

管理服务实例的创建和注入
"""

from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from kg_search.config import get_settings
from kg_search.extraction import GraphBuilder
from kg_search.graphrag import GraphRAGAdapter
from kg_search.indexing import ChromaVectorStore, DocumentStore, Neo4jGraphStore
from kg_search.ingestion import IngestionPipeline
from kg_search.llm import (
    AnswerGenerator,
    EmbeddingService,
    LLMClient,
    create_embedding_service,
    create_llm_client,
)
from kg_search.retrieval import GraphRetriever, HybridRetriever, VectorRetriever
from kg_search.retrieval.strategies import GlobalSearch, LocalSearch
from kg_search.utils import get_logger

logger = get_logger(__name__)

# 全局服务实例
_services: dict = {}

# API Key认证
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def init_services() -> None:
    """初始化所有服务"""
    global _services
    settings = get_settings()

    try:
        # LLM服务 (根据配置自动选择 OpenAI 或 GLM)
        llm_client = create_llm_client()
        embedding_service = create_embedding_service()

        # 存储服务
        vector_store = ChromaVectorStore()
        graph_store = Neo4jGraphStore()
        document_store = DocumentStore()

        # 连接Neo4j
        await graph_store.connect()
        await graph_store.init_schema()

        # 检索服务
        vector_retriever = VectorRetriever(vector_store, embedding_service)
        graph_retriever = GraphRetriever(graph_store)
        hybrid_retriever = HybridRetriever(vector_retriever, graph_retriever)

        # 检索策略
        local_search = LocalSearch(vector_retriever, graph_retriever, llm_client)
        global_search = GlobalSearch(llm_client)

        # 数据处理
        ingestion_pipeline = IngestionPipeline()
        graph_builder = GraphBuilder(llm_client)
        answer_generator = AnswerGenerator(llm_client)

        # GraphRAG适配器
        graphrag_adapter = GraphRAGAdapter(
            llm_client=llm_client,
            vector_retriever=vector_retriever,
            graph_store=graph_store,
        )

        _services = {
            "settings": settings,
            "llm_client": llm_client,
            "embedding_service": embedding_service,
            "vector_store": vector_store,
            "graph_store": graph_store,
            "document_store": document_store,
            "vector_retriever": vector_retriever,
            "graph_retriever": graph_retriever,
            "hybrid_retriever": hybrid_retriever,
            "local_search": local_search,
            "global_search": global_search,
            "ingestion_pipeline": ingestion_pipeline,
            "graph_builder": graph_builder,
            "answer_generator": answer_generator,
            "graphrag_adapter": graphrag_adapter,
        }

        logger.info("All services initialized successfully")

    except Exception as e:
        logger.error("Service initialization failed", error=str(e))
        raise


async def cleanup_services() -> None:
    """清理服务资源"""
    global _services

    if "graph_store" in _services:
        await _services["graph_store"].close()

    if "llm_client" in _services:
        await _services["llm_client"].close()

    if "embedding_service" in _services:
        await _services["embedding_service"].close()

    _services = {}
    logger.info("All services cleaned up")


# API Key验证
async def verify_api_key(
    api_key: str = Security(api_key_header),
) -> bool:
    """
    验证API Key

    Args:
        api_key: 请求头中的API Key

    Returns:
        验证是否通过

    Raises:
        HTTPException: 验证失败
    """
    settings = get_settings()

    # 如果未配置API Key，跳过验证（开发模式）
    if not settings.api_key:
        return True

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key",
        )

    return True


# 依赖注入函数
def get_llm_client() -> LLMClient:
    """获取LLM客户端"""
    return _services["llm_client"]


def get_embedding_service() -> EmbeddingService:
    """获取Embedding服务"""
    return _services["embedding_service"]


def get_vector_store() -> ChromaVectorStore:
    """获取向量存储"""
    return _services["vector_store"]


def get_graph_store() -> Neo4jGraphStore:
    """获取图存储"""
    return _services["graph_store"]


def get_document_store() -> DocumentStore:
    """获取文档存储"""
    return _services["document_store"]


def get_vector_retriever() -> VectorRetriever:
    """获取向量检索器"""
    return _services["vector_retriever"]


def get_graph_retriever() -> GraphRetriever:
    """获取图检索器"""
    return _services["graph_retriever"]


def get_hybrid_retriever() -> HybridRetriever:
    """获取混合检索器"""
    return _services["hybrid_retriever"]


def get_local_search() -> LocalSearch:
    """获取Local Search"""
    return _services["local_search"]


def get_global_search() -> GlobalSearch:
    """获取Global Search"""
    return _services["global_search"]


def get_ingestion_pipeline() -> IngestionPipeline:
    """获取提取管道"""
    return _services["ingestion_pipeline"]


def get_graph_builder() -> GraphBuilder:
    """获取图谱构建器"""
    return _services["graph_builder"]


def get_answer_generator() -> AnswerGenerator:
    """获取答案生成器"""
    return _services["answer_generator"]


def get_graphrag_adapter() -> GraphRAGAdapter:
    """获取GraphRAG适配器"""
    return _services["graphrag_adapter"]


# 类型别名
LLMClientDep = Annotated[LLMClient, Depends(get_llm_client)]
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]
VectorStoreDep = Annotated[ChromaVectorStore, Depends(get_vector_store)]
GraphStoreDep = Annotated[Neo4jGraphStore, Depends(get_graph_store)]
DocumentStoreDep = Annotated[DocumentStore, Depends(get_document_store)]
VectorRetrieverDep = Annotated[VectorRetriever, Depends(get_vector_retriever)]
GraphRetrieverDep = Annotated[GraphRetriever, Depends(get_graph_retriever)]
HybridRetrieverDep = Annotated[HybridRetriever, Depends(get_hybrid_retriever)]
LocalSearchDep = Annotated[LocalSearch, Depends(get_local_search)]
GlobalSearchDep = Annotated[GlobalSearch, Depends(get_global_search)]
IngestionPipelineDep = Annotated[IngestionPipeline, Depends(get_ingestion_pipeline)]
GraphBuilderDep = Annotated[GraphBuilder, Depends(get_graph_builder)]
AnswerGeneratorDep = Annotated[AnswerGenerator, Depends(get_answer_generator)]
GraphRAGAdapterDep = Annotated[GraphRAGAdapter, Depends(get_graphrag_adapter)]
ApiKeyDep = Annotated[bool, Depends(verify_api_key)]
