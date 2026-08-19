from fastapi import APIRouter
from common.logger import get_logger
from app.schemas.request import QueryReq
from fastapi.responses import StreamingResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/api/qa", tags=["RAG问答"])

@router.post("/query", summary="RAG问答查询")
def rag_query(req: QueryReq):
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
    )
    return ApiResponse(data=result)

@router.post("/stream", summary="流式RAG问答")
def rag_stream_query(req: QueryReq):
    from app.deps import rag_service
    import json
    from fastapi.responses import StreamingResponse

    final_prompt, filtered_results = rag_service.prepare_query_context(
        user_question=req.question,
        top_k=req.top_k,
        similarity_threshold=req.similarity_threshold,
        collection_name=req.collection_name,
        enable_rerank=req.enable_rerank,
        rerank_top_n=req.rerank_top_n
    )

    def stream_generator():
        try:
            # 空结果兜底
            if not filtered_results:
                yield f"data: {json.dumps({'type': 'content', 'data': '抱歉，暂无与该问题相关的政策信息。'}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'sources', 'data': []}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
                return

            # 流式输出大模型回答
            rag_service.llm_client.clear_history()
            for delta in rag_service.llm_client.chat_stream(final_prompt):
                # 标准SSE正文增量事件
                yield f"data: {json.dumps({'type': 'content', 'data': delta}, ensure_ascii=False)}\n\n"

            # 独立输出来源片段事件
            sources = [
                {
                    "content": doc["content"],
                    "score": float(doc.get("similarity", 0.0))
                }
                for doc in filtered_results
            ]
            yield f"data: {json.dumps({'type': 'sources', 'data': sources}, ensure_ascii=False)}\n\n"

            # 输出明确结束标记
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"流式问答执行异常: {str(e)}", exc_info=True)
            # 异常场景流式返回友好提示，不直接断开连接
            yield f"data: {json.dumps({'type': 'error', 'message': '服务异常，请稍后重试'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )