from fastapi import APIRouter

router = APIRouter(tags=["系统运维"])

@router.get("/health", summary="健康检查")
def health_check():
    from app.schemas.response import ApiResponse
    return ApiResponse(data={"status": "ok", "service": "gov-knowledge-rag"})