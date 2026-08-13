from typing import Optional
from pydantic import BaseModel, Field

class BuildKnowledgeReq(BaseModel):
    docs_dir: str = Field("./data/docs", min_length=1, description="待构建的本地文档目录路径")

class QueryReq(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="用户查询问题")
    top_k: int = Field(default=None, ge=1, le=20, description="召回片段数量，不传使用系统默认值")
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="相似度过滤阈值，0-1之间")
    return_sources: bool = Field(default=True, description="是否返回原文来源片段")
    collection_name: str = Field(
        default=None,
        pattern=r"^[a-zA-Z0-9_-]{2,64}$",
        description="目标知识库名称，不传使用默认集合，仅允许字母、数字、下划线、短横线"
    )
    
    enable_rerank: Optional[bool] = Field(default=None, description="是否启用重排序，不传则使用系统配置")
    rerank_top_n: Optional[int] = Field(default=None, ge=1, le=10, description="重排序返回的片段数量")

class AddDocumentReq(BaseModel):
    file_path: str = Field(..., min_length=1, description="待入库文档的本地绝对路径")
    collection_name: str = Field(
        default=None,
        pattern=r"^[a-zA-Z0-9_-]{2,64}$",
        description="目标知识库名称，不传使用默认集合，仅允许字母、数字、下划线、短横线"
    )