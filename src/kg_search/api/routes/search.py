"""
检索接口
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from kg_search.api.dependencies import (
    AnswerGeneratorDep,
    ApiKeyDep,
    GlobalSearchDep,
    GraphRetrieverDep,
    HybridRetrieverDep,
    LocalSearchDep,
    VectorRetrieverDep,
)
from kg_search.api.schemas.request import (
    GraphSearchRequest,
    QuestionRequest,
    SearchRequest,
    VectorSearchRequest,
)
from kg_search.api.schemas.response import (
    QuestionResponse,
    SearchResponse,
    SearchResult,
)
from kg_search.utils import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def hybrid_search(
    request: SearchRequest,
    _: ApiKeyDep,
    hybrid_retriever: HybridRetrieverDep,
):
    """
    混合检索接口

    结合向量检索和图检索，支持多种融合策略
    """
    try:
        results = await hybrid_retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            entity_types=request.entity_types,
            fusion_method=request.fusion_method,
        )

        return SearchResponse(
            query=request.query,
            results=[
                SearchResult(
                    id=r.id,
                    content=r.content,
                    score=r.score,
                    metadata=r.metadata,
                    source_type=r.source_type,
                )
                for r in results
            ],
            total=len(results),
        )

    except Exception as e:
        logger.error("Hybrid search failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")


@router.post("/search/vector", response_model=SearchResponse)
async def vector_search(
    request: VectorSearchRequest,
    _: ApiKeyDep,
    vector_retriever: VectorRetrieverDep,
):
    """
    向量检索接口

    基于语义相似度的检索
    """
    try:
        results = await vector_retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            min_score=request.min_score,
        )

        return SearchResponse(
            query=request.query,
            results=[
                SearchResult(
                    id=r.id,
                    content=r.content,
                    score=r.score,
                    metadata=r.metadata,
                    source_type="vector",
                )
                for r in results
            ],
            total=len(results),
        )

    except Exception as e:
        logger.error("Vector search failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"向量检索失败: {str(e)}")


@router.post("/search/graph", response_model=SearchResponse)
async def graph_search(
    request: GraphSearchRequest,
    _: ApiKeyDep,
    graph_retriever: GraphRetrieverDep,
):
    """
    图检索接口

    基于知识图谱的关联检索
    """
    try:
        if request.entity_name:
            # 基于实体的检索
            results = await graph_retriever.retrieve_by_entity(
                entity_name=request.entity_name,
                relation_types=request.relation_types,
                depth=request.depth,
                limit=request.top_k,
            )
        else:
            # 基于查询的检索
            results = await graph_retriever.retrieve_by_query(
                query=request.query,
                entity_types=request.entity_types,
                limit=request.top_k,
            )

        return SearchResponse(
            query=request.query,
            results=[
                SearchResult(
                    id=r.id,
                    content=r.content,
                    score=r.score,
                    metadata=r.metadata,
                    source_type="graph",
                )
                for r in results
            ],
            total=len(results),
        )

    except Exception as e:
        logger.error("Graph search failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"图检索失败: {str(e)}")


@router.post("/search/local", response_model=QuestionResponse)
async def local_search(
    request: QuestionRequest,
    _: ApiKeyDep,
    local_search: LocalSearchDep,
):
    """
    GraphRAG Local Search

    适用于具体问题，如：
    - "四羊方尊是什么朝代的？"
    - "这件文物的材质是什么？"
    """
    try:
        if request.generate_answer:
            result = await local_search.search_with_answer(
                query=request.question,
                top_k=request.top_k,
            )
        else:
            result = await local_search.search(
                query=request.question,
                top_k=request.top_k,
                include_neighbors=True,
            )
            result["answer"] = None

        return QuestionResponse(
            question=request.question,
            answer=result.get("answer"),
            context=result.get("context", ""),
            sources=[
                SearchResult(
                    id=c["id"],
                    content=c["content"],
                    score=c["score"],
                    metadata=c.get("metadata", {}),
                    source_type=c.get("source_type", "local"),
                )
                for c in result.get("chunks", [])
            ],
        )

    except Exception as e:
        logger.error("Local search failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Local Search失败: {str(e)}")


@router.post("/search/global", response_model=QuestionResponse)
async def global_search(
    request: QuestionRequest,
    _: ApiKeyDep,
    global_search: GlobalSearchDep,
):
    """
    GraphRAG Global Search

    适用于宏观问题，如：
    - "中国青铜器的发展历程是怎样的？"
    - "不同朝代的玉器有什么特点？"
    """
    try:
        result = await global_search.search(
            query=request.question,
            top_k=request.top_k,
        )

        return QuestionResponse(
            question=request.question,
            answer=result.get("answer"),
            context=result.get("context", ""),
            sources=[
                SearchResult(
                    id=c["id"],
                    content=c.get("summary", ""),
                    score=0.0,
                    metadata={"title": c.get("title", "")},
                    source_type="community",
                )
                for c in result.get("communities", [])
            ],
        )

    except Exception as e:
        logger.error("Global search failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Global Search失败: {str(e)}")


@router.post("/ask")
async def ask_question(
    request: QuestionRequest,
    _: ApiKeyDep,
    hybrid_retriever: HybridRetrieverDep,
    answer_generator: AnswerGeneratorDep,
):
    """
    智能问答接口

    自动选择检索策略并生成答案
    """
    try:
        # 检索
        results = await hybrid_retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
        )

        # 构建上下文
        context = "\n\n".join([r.content for r in results])

        # 生成答案
        if request.stream:
            # 流式响应
            async def generate():
                async for chunk in answer_generator.generate_stream(
                    query=request.question,
                    context=context,
                ):
                    yield chunk

            return StreamingResponse(
                generate(),
                media_type="text/plain",
            )
        else:
            # 非流式响应
            if request.generate_answer:
                response = await answer_generator.generate_with_citations(
                    query=request.question,
                    retrieval_results=[r.to_dict() for r in results],
                )
                answer = response["answer"]
                sources = response["sources"]
            else:
                answer = None
                sources = [r.to_dict() for r in results]

            return QuestionResponse(
                question=request.question,
                answer=answer,
                context=context,
                sources=[
                    SearchResult(
                        id=s.get("source_id", s.get("id", "")),
                        content=s.get("content_preview", s.get("content", "")),
                        score=s.get("score", 0.0),
                        metadata=s.get("metadata", {}),
                        source_type="hybrid",
                    )
                    for s in sources
                ],
            )

    except Exception as e:
        logger.error("Question answering failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")
