"""
RAG 问答路由模块
提供同步查询与流式（SSE）两类问答接口，统一走 RagService 完成检索、重排与生成全链路
依赖：RagService、QueryReq、StreamingResponse
"""
from fastapi import APIRouter
from app.common.logger import get_logger
from app.schemas.request import QueryReq
from fastapi.responses import StreamingResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/api/qa", tags=["RAG问答"])

@router.post("/query", summary="RAG问答查询")
def rag_query(req: QueryReq):
    """同步 RAG 问答接口，一次性返回答案与来源片段"""
    from app.deps import rag_service
    from app.schemas.response import ApiResponse
    result = rag_service.query(
        user_question=req.question,
        top_k=req.top_k,
        similarity_threshold=req.similarity_threshold,
        return_sources=req.return_sources,
        collection_name=req.collection_name,
        enable_rerank=req.enable_rerank,
        rerank_top_n=req.rerank_top_n,
        session_id=req.session_id
    )
    return ApiResponse(data=result)

@router.post("/stream", summary="流式RAG问答")
def rag_stream_query(req: QueryReq):
    """流式 RAG 问答接口，以 SSE 事件流逐块推送 content/sources/done 事件"""
    from app.deps import rag_service
    import json

    def event_generator():
        for event in rag_service.stream_query(
            user_question=req.question,
            top_k=req.top_k,
            similarity_threshold=req.similarity_threshold,
            return_sources=req.return_sources,
            collection_name=req.collection_name,
            enable_rerank=req.enable_rerank,
            rerank_top_n=req.rerank_top_n,
            session_id=req.session_id
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

