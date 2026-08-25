"""
健康检查路由模块
对外提供系统存活探针接口，供运维监控与负载均衡探测服务可用性
依赖：FastAPI APIRouter、app.schemas.response.ApiResponse
"""
from fastapi import APIRouter

router = APIRouter(tags=["系统运维"])

@router.get("/health", summary="健康检查")
def health_check():
    """健康检查接口，返回服务名与运行状态标识"""
    from app.schemas.response import ApiResponse
    return ApiResponse(data={"status": "ok", "service": "gov-knowledge-rag"})