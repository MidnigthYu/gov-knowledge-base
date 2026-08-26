"""
请求数据模型模块
定义知识库构建、RAG 问答、单文档入库三类接口的请求体结构，统一入参校验规则
依赖：Pydantic BaseModel/Field
"""
from typing import Optional
from pydantic import BaseModel, Field

class BuildKnowledgeReq(BaseModel):
    """知识库批量构建请求体，指定源目录与目标集合"""
    docs_dir: str = Field("./data/docs", min_length=1, description="待构建的本地文档目录路径")
    collection_name: str = Field(
        default=None,
        pattern=r"^[a-zA-Z0-9_-]{2,64}$",
        description="目标知识库名称，不传使用默认集合，仅允许字母、数字、下划线、短横线"
    )

class QueryReq(BaseModel):
    """RAG 问答请求体，携带检索、重排序与会话相关配置"""
    question: str = Field(..., min_length=1, max_length=500, description="用户查询问题")
    top_k: int = Field(default=None, ge=1, le=20, description="召回片段数量，不传使用系统默认值")
    similarity_threshold: float = Field(default=None, ge=0.0, le=1.0, description="相似度过滤阈值，不传使用系统默认值，0-1之间")
    return_sources: bool = Field(default=True, description="是否返回原文来源片段")
    session_id: Optional[str] = Field(
        default=None,
        description="会话ID，用于多轮上下文记忆，不传则为单轮问答，会话有效期30分钟"
    )
    collection_name: str = Field(
        default=None,
        pattern=r"^[a-zA-Z0-9_-]{2,64}$",
        description="目标知识库名称，不传使用默认集合，仅允许字母、数字、下划线、短横线"
    )
    
    enable_rerank: Optional[bool] = Field(default=None, description="是否启用重排序，不传则使用系统配置")
    rerank_top_n: Optional[int] = Field(default=None, ge=1, le=10, description="重排序返回的片段数量")

class AddDocumentReq(BaseModel):
    """单文档增量入库请求体，指定本地文档绝对路径与目标集合"""
    file_path: str = Field(..., min_length=1, description="待入库文档的本地绝对路径")
    collection_name: str = Field(
        default=None,
        pattern=r"^[a-zA-Z0-9_-]{2,64}$",
        description="目标知识库名称，不传使用默认集合，仅允许字母、数字、下划线、短横线"
    )