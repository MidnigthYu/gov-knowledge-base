"""API 路由包，承载健康检查、知识库管理、RAG 问答等路由模块"""

from .health import router as health_router
from .knowledge import router as knowledge_router
from .qa import router as qa_router

__all__ = ["health_router", "knowledge_router", "qa_router"]

