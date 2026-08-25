"""
统一响应数据模型
所有接口返回结构统一为 {code, message, data}，与 ResponseUtil 的返回格式保持一致
依赖：Pydantic BaseModel
"""
from pydantic import BaseModel
from typing import Optional

class ApiResponse(BaseModel):
    """标准接口响应体，code=0 表示成功，data 承载业务载荷"""
    code: int = 0
    message: str = "success"
    data: Optional[dict] = None