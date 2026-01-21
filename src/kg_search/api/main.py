"""
FastAPI应用入口
"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from kg_search.config import get_settings
from kg_search.utils import get_logger, setup_logging

from .dependencies import cleanup_services, init_services
from .routes import health, ingest, search

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    settings = get_settings()

    # 启动时初始化
    setup_logging(log_level=settings.log_level)
    logger.info("Starting KG-Search API", version=settings.app_version)

    # 初始化服务
    await init_services()

    yield

    # 关闭时清理
    await cleanup_services()
    logger.info("KG-Search API shutdown complete")


# 创建FastAPI应用
app = FastAPI(
    title="KG-Search API",
    description="文博领域数字文物库智能检索系统 - 基于GraphRAG + 向量相似度的混合检索",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(health.router, tags=["健康检查"])
app.include_router(ingest.router, prefix="/api/v1", tags=["数据提取"])
app.include_router(search.router, prefix="/api/v1", tags=["检索服务"])


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "KG-Search API",
        "version": "0.1.0",
        "description": "文博领域数字文物库智能检索系统",
        "docs": "/docs",
    }


def run():
    """运行API服务"""
    settings = get_settings()
    uvicorn.run(
        "kg_search.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    run()
