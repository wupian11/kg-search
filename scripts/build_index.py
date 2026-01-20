#!/usr/bin/env python
"""
索引构建脚本

从数据目录读取文件，构建向量索引和知识图谱
"""

import argparse
import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kg_search.config import get_settings
from kg_search.utils import setup_logging, get_logger
from kg_search.ingestion import IngestionPipeline
from kg_search.extraction import GraphBuilder
from kg_search.indexing import ChromaVectorStore, Neo4jGraphStore, DocumentStore
from kg_search.llm import OpenAIClient, OpenAIEmbedding
from kg_search.retrieval.strategies import GlobalSearch

logger = get_logger(__name__)


async def build_index(
    data_dir: str,
    build_graph: bool = True,
    build_communities: bool = True,
):
    """
    构建索引
    
    Args:
        data_dir: 数据目录
        build_graph: 是否构建知识图谱
        build_communities: 是否构建社区（Global Search）
    """
    settings = get_settings()
    setup_logging(log_level=settings.log_level)
    
    logger.info("Starting index build", data_dir=data_dir)
    
    # 初始化服务
    llm_client = OpenAIClient()
    embedding_service = OpenAIEmbedding()
    vector_store = ChromaVectorStore()
    graph_store = Neo4jGraphStore()
    document_store = DocumentStore()
    
    pipeline = IngestionPipeline()
    graph_builder = GraphBuilder(llm_client)
    
    try:
        # 连接Neo4j
        await graph_store.connect()
        
        # 处理数据目录
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.error("Data directory not found", path=data_dir)
            return
        
        logger.info("Processing data directory...")
        documents, chunks = pipeline.process_directory(data_path)
        
        if not documents:
            logger.warning("No documents found in data directory")
            return
        
        logger.info(f"Found {len(documents)} documents, {len(chunks)} chunks")
        
        # 保存文档
        logger.info("Saving documents...")
        await document_store.add_documents(documents)
        await document_store.add_chunks(chunks)
        
        # 生成向量
        logger.info("Generating embeddings...")
        texts = [chunk.content for chunk in chunks]
        embeddings = await embedding_service.embed_texts(texts)
        
        # 存储向量
        logger.info("Storing vectors...")
        await vector_store.add_chunks(chunks, embeddings)
        
        # 构建知识图谱
        if build_graph:
            logger.info("Building knowledge graph...")
            kg = await graph_builder.build_from_documents(documents)
            
            logger.info(
                "Knowledge graph built",
                entities=len(kg.entities),
                relations=len(kg.relations),
            )
            
            # 存储到Neo4j
            logger.info("Storing graph to Neo4j...")
            await graph_store.add_knowledge_graph(kg)
            
            # 构建社区
            if build_communities:
                logger.info("Building communities for Global Search...")
                global_search = GlobalSearch(llm_client)
                communities = await global_search.build_communities(kg)
                
                # 保存社区数据
                communities_file = Path(settings.processed_data_dir) / "communities.json"
                global_search.save_communities(str(communities_file))
                
                logger.info(f"Built {len(communities)} communities")
        
        logger.info("Index build completed successfully!")
        
        # 打印统计
        print("\n" + "="*50)
        print("索引构建完成!")
        print("="*50)
        print(f"文档数量: {len(documents)}")
        print(f"文本块数量: {len(chunks)}")
        print(f"向量数量: {vector_store.get_collection_stats()['count']}")
        if build_graph:
            print(f"实体数量: {len(kg.entities)}")
            print(f"关系数量: {len(kg.relations)}")
            if build_communities:
                print(f"社区数量: {len(communities)}")
        print("="*50)
        
    except Exception as e:
        logger.error("Index build failed", error=str(e))
        raise
    
    finally:
        await graph_store.close()
        await llm_client.close()
        await embedding_service.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="构建知识索引")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/raw",
        help="数据目录路径",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="不构建知识图谱",
    )
    parser.add_argument(
        "--no-communities",
        action="store_true",
        help="不构建社区（跳过Global Search索引）",
    )
    
    args = parser.parse_args()
    
    asyncio.run(build_index(
        data_dir=args.data_dir,
        build_graph=not args.no_graph,
        build_communities=not args.no_communities,
    ))


if __name__ == "__main__":
    main()
