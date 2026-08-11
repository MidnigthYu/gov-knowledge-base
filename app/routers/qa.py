from fastapi import APIRouter
from app.schemas.request import QueryReq

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
        collection_name=req.collection_name
    )
    return ApiResponse(data=result)