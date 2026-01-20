#!/usr/bin/env python
"""
Neo4j数据库初始化脚本

创建必要的索引和约束
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kg_search.config import get_settings
from kg_search.indexing import Neo4jGraphStore
from kg_search.utils import setup_logging, get_logger

logger = get_logger(__name__)


async def init_neo4j():
    """初始化Neo4j数据库"""
    settings = get_settings()
    setup_logging(log_level=settings.log_level)
    
    logger.info("Initializing Neo4j database...")
    
    graph_store = Neo4jGraphStore()
    
    try:
        await graph_store.connect()
        await graph_store.init_schema()
        logger.info("Neo4j initialization completed successfully")
        
    except Exception as e:
        logger.error("Neo4j initialization failed", error=str(e))
        raise
    
    finally:
        await graph_store.close()


def main():
    """主函数"""
    asyncio.run(init_neo4j())


if __name__ == "__main__":
    main()
