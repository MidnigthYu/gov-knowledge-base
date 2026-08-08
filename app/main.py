from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config.settings import settings
from client.zhipu_client import ZhipuClient
from client.zhipu_embedding_client import ZhipuEmbeddingClient
from vector_store.chroma_store import ChromaVectorStore
from service.knowledge_manager import KnowledgeManager
from service.rag_service import RagService
from common.exceptions import GovRAGBaseError

# ========== 全局服务实例（服务启动时初始化1次） ==========
_llm_client = None
_embedding_client = None
_vector_store = None
_kb_manager = None
_rag_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务生命周期：启动时初始化业务组件"""
    global _llm_client, _embedding_client, _vector_store
    global _kb_manager, _rag_service

    _llm_client = ZhipuClient()
    _embedding_client = ZhipuEmbeddingClient()
    _vector_store = ChromaVectorStore(
    collection_name=settings.CHROMA_DEFAULT_COLLECTION,
    persist_dir=settings.CHROMA_PERSIST_DIR
)
    _kb_manager = KnowledgeManager(
        embedding_client=_embedding_client,
        vector_store=_vector_store
    )
    _rag_service = RagService(
        vector_store=_vector_store,
        llm_client=_llm_client,
        embedding_client=_embedding_client
    )
    yield


# 初始化 FastAPI 应用，绑定生命周期（关键修正）
app = FastAPI(
    title="政务知识库 RAG 系统",
    version="1.0.0",
    lifespan=lifespan
)

# ========== 统一响应模型 ==========
class ApiResponse(BaseModel):
    code: int = 0
    message: str = "success"
    data: dict = None


# ========== 请求体模型 ==========
class BuildKnowledgeReq(BaseModel):
    docs_dir: str = "./data/docs"


class QueryReq(BaseModel):
    question: str


# ========== 全局异常处理 ==========
@app.exception_handler(GovRAGBaseError)
def handle_business_exception(request, exc: GovRAGBaseError):
    return ApiResponse(code=500, message=str(exc), data=None)


@app.exception_handler(Exception)
def handle_unknown_exception(request, exc: Exception):
    return ApiResponse(code=500, message="服务内部错误", data=None)


# ========== 业务接口 ==========
@app.get("/health", summary="健康检查")
def health_check():
    return ApiResponse(data={"status": "ok", "service": "gov-knowledge-rag"})


@app.post("/api/knowledge/build", summary="批量构建知识库")
def build_knowledge(req: BuildKnowledgeReq):
    try:
        result = _kb_manager.build_from_dir(req.docs_dir)
        return ApiResponse(data=result)
    except GovRAGBaseError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/qa/query", summary="RAG 问答查询")
def rag_query(req: QueryReq):
    if not req.question.strip():
        return ApiResponse(code=400, message="问题不能为空", data=None)
    try:
        result = _rag_service.query(req.question)
        return ApiResponse(data=result)
    except GovRAGBaseError as e:
        raise HTTPException(status_code=500, detail=str(e))