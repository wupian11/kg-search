"""
数据摄入接口
"""

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from kg_search.api.dependencies import (
    ApiKeyDep,
    DocumentStoreDep,
    EmbeddingServiceDep,
    GraphBuilderDep,
    GraphStoreDep,
    IngestionPipelineDep,
    VectorStoreDep,
)
from kg_search.api.schemas.request import IngestFileRequest, IngestTextRequest
from kg_search.api.schemas.response import IngestResponse, TaskResponse
from kg_search.utils import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(
    request: IngestTextRequest,
    _: ApiKeyDep,
    pipeline: IngestionPipelineDep,
    graph_builder: GraphBuilderDep,
    embedding_service: EmbeddingServiceDep,
    vector_store: VectorStoreDep,
    graph_store: GraphStoreDep,
    document_store: DocumentStoreDep,
):
    """
    摄入文本数据

    支持的格式：json, jsonl, md, txt
    """
    try:
        # 1. 解析和分块
        documents, chunks = pipeline.process_text(
            request.content,
            doc_type=request.doc_type,
            metadata=request.metadata,
        )

        # 2. 保存文档和块
        await document_store.add_documents(documents)
        await document_store.add_chunks(chunks)

        # 3. 生成向量并存储
        if chunks:
            texts = [chunk.content for chunk in chunks]
            embeddings = await embedding_service.embed_texts(texts)
            await vector_store.add_chunks(chunks, embeddings)

        # 4. 构建知识图谱
        if request.build_graph and documents:
            kg = await graph_builder.build_from_documents(documents)
            await graph_store.add_knowledge_graph(kg)
            graph_stats = {
                "entities": len(kg.entities),
                "relations": len(kg.relations),
            }
        else:
            graph_stats = None

        return IngestResponse(
            success=True,
            message="数据摄入成功",
            documents_count=len(documents),
            chunks_count=len(chunks),
            graph_stats=graph_stats,
        )

    except Exception as e:
        logger.error("Text ingestion failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"摄入失败: {str(e)}")


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    build_graph: bool = Form(True),
    _: ApiKeyDep = None,
    pipeline: IngestionPipelineDep = None,
    graph_builder: GraphBuilderDep = None,
    embedding_service: EmbeddingServiceDep = None,
    vector_store: VectorStoreDep = None,
    graph_store: GraphStoreDep = None,
    document_store: DocumentStoreDep = None,
):
    """
    摄入文件数据

    支持的文件格式：.json, .jsonl, .md, .txt
    """
    # 检查文件类型
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    suffix = Path(file.filename).suffix.lower()
    supported = [".json", ".jsonl", ".md", ".txt"]

    if suffix not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {suffix}，支持的格式: {', '.join(supported)}",
        )

    try:
        # 读取文件内容
        content = await file.read()
        text = content.decode("utf-8")

        # 确定文档类型
        doc_type_map = {
            ".json": "json",
            ".jsonl": "jsonl",
            ".md": "md",
            ".txt": "txt",
        }
        doc_type = doc_type_map[suffix]

        # 处理
        documents, chunks = pipeline.process_text(
            text,
            doc_type=doc_type,
            metadata={"source_file": file.filename},
        )

        # 保存
        await document_store.add_documents(documents)
        await document_store.add_chunks(chunks)

        # 向量化
        if chunks:
            texts = [chunk.content for chunk in chunks]
            embeddings = await embedding_service.embed_texts(texts)
            await vector_store.add_chunks(chunks, embeddings)

        # 构建图谱
        if build_graph and documents:
            kg = await graph_builder.build_from_documents(documents)
            await graph_store.add_knowledge_graph(kg)
            graph_stats = {
                "entities": len(kg.entities),
                "relations": len(kg.relations),
            }
        else:
            graph_stats = None

        return IngestResponse(
            success=True,
            message=f"文件 {file.filename} 摄入成功",
            documents_count=len(documents),
            chunks_count=len(chunks),
            graph_stats=graph_stats,
        )

    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="文件编码错误，请使用UTF-8编码")
    except Exception as e:
        logger.error("File ingestion failed", error=str(e), filename=file.filename)
        raise HTTPException(status_code=500, detail=f"摄入失败: {str(e)}")


@router.post("/ingest/directory", response_model=TaskResponse)
async def ingest_directory(
    request: IngestFileRequest,
    background_tasks: BackgroundTasks,
    _: ApiKeyDep,
):
    """
    异步摄入目录数据（后台任务）

    用于大量数据的批量摄入
    """
    # TODO: 实现后台任务处理
    return TaskResponse(
        task_id="not_implemented",
        status="pending",
        message="目录摄入功能开发中",
    )


@router.delete("/ingest/document/{doc_id}")
async def delete_document(
    doc_id: str,
    _: ApiKeyDep,
    document_store: DocumentStoreDep,
    vector_store: VectorStoreDep,
):
    """删除文档及其相关数据"""
    try:
        # 获取文档相关的块ID
        chunks = await document_store.get_chunks_by_document(doc_id)
        chunk_ids = [chunk.id for chunk in chunks]

        # 删除向量
        if chunk_ids:
            await vector_store.delete(chunk_ids)

        # 删除文档
        success = await document_store.delete_document(doc_id)

        if success:
            return {"success": True, "message": f"文档 {doc_id} 已删除"}
        else:
            raise HTTPException(status_code=404, detail=f"文档 {doc_id} 不存在")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Document deletion failed", error=str(e), doc_id=doc_id)
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")
