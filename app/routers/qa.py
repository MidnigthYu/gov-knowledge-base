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

    final_system_prompt, filtered_results = rag_service.prepare_query_context(
        user_question=req.question,
        top_k=req.top_k,
        similarity_threshold=req.similarity_threshold,
        collection_name=req.collection_name,
        enable_rerank=req.enable_rerank,
        rerank_top_n=req.rerank_top_n,
        session_id=req.session_id
    )

    def stream_generator():
        """SSE 事件生成器，逐块产出 content/sources/done 事件，异常时降级为 error 事件"""
        full_answer = ""
        try:
            # 空结果兜底
            if not filtered_results:
                empty_answer = "抱歉，暂无与该问题相关的政策信息。"
                yield f"data: {json.dumps({'type': 'content', 'data': empty_answer}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'sources', 'data': []}, ensure_ascii=False)}\n\n"
                # 空回答也保存历史，保证上下文连续
                rag_service._append_session_history(req.session_id, req.question, empty_answer)
                yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
                return

            rag_service.llm_client.clear_history()
            rag_service.llm_client.messages[0]["content"] = final_system_prompt
            
            # 流式生成回答
            for delta in rag_service.llm_client.chat_stream(prompt=req.question):
                full_answer += delta
                yield f"data: {json.dumps({'type': 'content', 'data': delta}, ensure_ascii=False)}\n\n"

            # 流式生成完成后，保存本轮完整对话到会话历史
            rag_service._append_session_history(req.session_id, req.question, full_answer)

            # 输出来源片段
            sources = [
                {
                    "content": doc["content"],
                    "score": float(doc.get("similarity", 0.0))
                }
                for doc in filtered_results
            ]
            yield f"data: {json.dumps({'type': 'sources', 'data': sources}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"流式问答执行异常: {str(e)}", exc_info=True)
            # 异常分支补存已生成的部分回答，避免历史断档
            if full_answer:
                rag_service._append_session_history(req.session_id, req.question, full_answer)
            yield f"data: {json.dumps({'type': 'error', 'message': '服务异常，请稍后重试'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")
