from app.routers import health, knowledge, qa
from app import deps
from contextlib import asynccontextmanager
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config.settings import settings
from client.zhipu_client import ZhipuClient
from client.zhipu_embedding_client import ZhipuEmbeddingClient
from vector_store.chroma_store import ChromaVectorStore
from service.knowledge_manager import KnowledgeManager
from service.rag_service import RagService
from common.exceptions import GovRAGBaseError, ErrorCode
from common.logger import get_logger

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务生命周期：启动时初始化业务组件"""
    deps.llm_client = ZhipuClient()
    deps.embedding_client = ZhipuEmbeddingClient()
    deps.vector_store = ChromaVectorStore(
        collection_name=settings.CHROMA_DEFAULT_COLLECTION,
        persist_dir=settings.CHROMA_PERSIST_DIR
    )
    deps.kb_manager = KnowledgeManager(
        embedding_client=deps.embedding_client,
        vector_store=deps.vector_store
    )
    deps.rag_service = RagService(
        vector_store=deps.vector_store,
        llm_client=deps.llm_client,
        embedding_client=deps.embedding_client
    )
    yield
    
app = FastAPI(
    title="政务知识库 RAG 系统",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(health.router)
app.include_router(knowledge.router)
app.include_router(qa.router)

# 全局异常处理
@app.exception_handler(GovRAGBaseError)
async def handle_business_exception(request: Request, exc: GovRAGBaseError):
    """统一捕获业务异常，返回标准错误码"""
    return JSONResponse(
        status_code=200,
        content={
            "code": exc.code,
            "message": exc.message,
            "data": None
        }
    )

@app.exception_handler(StarletteHTTPException)
async def handle_http_exception(request: Request, exc: StarletteHTTPException):
    """统一处理 Starlette HTTP 异常"""
    builtin_paths = ["/docs", "/redoc", "/openapi.json", "/openapi"]
    if any(request.url.path.startswith(p) for p in builtin_paths):
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.status_code, "message": exc.detail, "data": None}
        )

    if exc.status_code == 404:
        return JSONResponse(
            status_code=404,
            content={
                "code": 404,
                "message": "URL拼写可能存在错误，请检查",
                "data": None
            }
        )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.status_code,
            "message": exc.detail,
            "data": None
        }
    )

@app.exception_handler(Exception)
async def handle_unknown_exception(request: Request, exc: Exception):
    """捕获未处理的系统异常，隐藏堆栈信息"""
    logger.error(f"系统未捕获异常: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "code": ErrorCode.SYSTEM_ERROR.code,
            "message": ErrorCode.SYSTEM_ERROR.message,
            "data": None
        }
    )