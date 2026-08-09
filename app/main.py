from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config.settings import settings
from client.zhipu_client import ZhipuClient
from client.zhipu_embedding_client import ZhipuEmbeddingClient
from vector_store.chroma_store import ChromaVectorStore
from service.knowledge_manager import KnowledgeManager
from service.rag_service import RagService
from common.exceptions import GovRAGBaseError, ErrorCode
from common.logger import get_logger

logger = get_logger(__name__)

# 全局服务实例
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

# 初始化 FastAPI 应用
app = FastAPI(
    title="政务知识库 RAG 系统",
    version="1.0.0",
    lifespan=lifespan
)

# 统一响应模型
class ApiResponse(BaseModel):
    code: int = 0
    message: str = "success"
    data: dict = None

# 请求体模型
class BuildKnowledgeReq(BaseModel):
    docs_dir: str = "./data/docs"

class QueryReq(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="用户查询问题")
    top_k: int = Field(default=None, ge=1, le=20, description="召回片段数量，不传使用系统默认值")
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="相似度过滤阈值，0-1之间")
    return_sources: bool = Field(default=True, description="是否返回原文来源片段")

class AddDocumentReq(BaseModel):
    file_path: str = Field(..., min_length=1, description="待入库文档的本地绝对路径")
    collection_name: str = Field(default=None, description="目标知识库名称，不传使用默认集合")

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

# 业务接口
@app.get("/health", summary="健康检查")
def health_check():
    return ApiResponse(data={"status": "ok", "service": "gov-knowledge-rag"})

@app.post("/api/knowledge/build", summary="批量构建知识库")
def build_knowledge(req: BuildKnowledgeReq):
    result = _kb_manager.build_from_dir(req.docs_dir)
    return ApiResponse(data=result)

@app.post("/api/qa/query", summary="RAG问答查询")
def rag_query(req: QueryReq):
    result = _rag_service.query(
        user_question=req.question,
        top_k=req.top_k,
        similarity_threshold=req.similarity_threshold,
        return_sources=req.return_sources
    )
    return ApiResponse(data=result)

@app.get("/api/knowledge/list", summary="获取所有知识库列表")
def list_knowledge_bases():
    collections = _kb_manager.list_knowledge_bases()
    return ApiResponse(data={"collections": collections})

@app.delete("/api/knowledge/{collection_name}", summary="删除指定知识库")
def delete_knowledge_base(collection_name: str):
    _kb_manager.delete_knowledge_base(collection_name)
    return ApiResponse(data={"deleted_collection": collection_name})

@app.post("/api/knowledge/add-document", summary="单文档增量入库")
def add_single_document(req: AddDocumentReq):
    chunk_count = _kb_manager.add_single_document(
        file_path=req.file_path,
        collection_name=req.collection_name
    )
    return ApiResponse(data={"added_chunks": chunk_count})