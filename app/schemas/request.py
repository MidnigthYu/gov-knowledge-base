from pydantic import BaseModel, Field

class BuildKnowledgeReq(BaseModel):
    docs_dir: str = "./data/docs"

class QueryReq(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="用户查询问题")
    top_k: int = Field(default=None, ge=1, le=20, description="召回片段数量，不传使用系统默认值")
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="相似度过滤阈值，0-1之间")
    return_sources: bool = Field(default=True, description="是否返回原文来源片段")
    collection_name: str = Field(default=None, description="目标知识库名称，不传使用默认集合")
    
class AddDocumentReq(BaseModel):
    file_path: str = Field(..., min_length=1, description="待入库文档的本地绝对路径")
    collection_name: str = Field(default=None, description="目标知识库名称，不传使用默认集合")