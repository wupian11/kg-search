"""
健康检查接口
"""

from fastapi import APIRouter

from kg_search.api.dependencies import DocumentStoreDep, VectorStoreDep
from kg_search.config import get_settings

router = APIRouter()


@router.get("/health")
async def health_check():
    """基础健康检查"""
    return {"status": "healthy"}


@router.get("/health/ready")
async def readiness_check(
    vector_store: VectorStoreDep,
    document_store: DocumentStoreDep,
):
    """就绪检查（包含依赖服务）"""
    settings = get_settings()

    # 检查各服务状态
    checks = {
        "api": "healthy",
        "vector_store": "unknown",
        "document_store": "unknown",
    }

    try:
        # 检查向量存储
        stats = vector_store.get_collection_stats()
        checks["vector_store"] = "healthy"
        checks["vector_store_count"] = stats["count"]
    except Exception as e:
        checks["vector_store"] = f"unhealthy: {str(e)}"

    try:
        # 检查文档存储
        stats = document_store.get_stats()
        checks["document_store"] = "healthy"
        checks["document_count"] = stats["total_documents"]
        checks["chunk_count"] = stats["total_chunks"]
    except Exception as e:
        checks["document_store"] = f"unhealthy: {str(e)}"

    # 总体状态
    all_healthy = all(
        v == "healthy" for k, v in checks.items() if k in ["api", "vector_store", "document_store"]
    )

    return {
        "status": "ready" if all_healthy else "degraded",
        "version": settings.app_version,
        "checks": checks,
    }


@router.get("/health/live")
async def liveness_check():
    """存活检查"""
    return {"status": "alive"}
