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
        """失败响应，传入ErrorCode枚举成员"""
        message = error_enum.message
        if detail:
            message = f"{message}：{detail}"
        return {
            "code": error_enum.code,
            "message": message,
            "data": None
        }

    @staticmethod
    def error_by_code(code: int, message: str, detail: str = None) -> dict:
        """直接通过错误码和消息返回错误响应，适配自定义异常类"""
        if detail:
            message = f"{message}：{detail}"
        return {
            "code": code,
            "message": message,
            "data": None
        }