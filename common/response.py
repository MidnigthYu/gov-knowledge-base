"""
统一响应工具类
所有接口返回结构统一为 {code, message, data}
"""
from typing import Any, Optional
from common.exceptions import ErrorCode

class ResponseUtil:
    @staticmethod
    def success(data: Any = None, message: str = "success") -> dict:
        """成功响应"""
        return {
            "code": 0,
            "message": message,
            "data": data
        }

    @staticmethod
    def error(error_enum: ErrorCode, detail: Optional[str] = None) -> dict:
        """失败响应"""
        result = {
            "code": error_enum.code,
            "message": error_enum.message,
            "data": None
        }
        if detail:
            result["detail"] = detail
        return result