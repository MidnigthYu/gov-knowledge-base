"""
统一响应工具类
所有接口返回结构统一为 {code, message, data}
"""
from typing import Any, Optional
from app.common.exceptions import ErrorCode

class ResponseUtil:
    """统一响应构造工具，提供成功/失败两类标准响应结构的静态构造方法"""

    @staticmethod
    def success(data: Any = None, message: str = "success") -> dict:
        """构造成功响应体

        Args:
            data: 业务载荷数据，缺省为 None
            message: 成功提示信息，缺省为 "success"

        Returns:
            标准成功响应字典，code 固定为 0
        """
        return {
            "code": 0,
            "message": message,
            "data": data
        }

    @staticmethod
    def error(error_enum: ErrorCode, detail: Optional[str] = None) -> dict:
        """构造失败响应体

        Args:
            error_enum: 业务错误码枚举
            detail: 可选错误详情，透传到响应体的 detail 字段

        Returns:
            标准失败响应字典，code/message 取自错误码枚举
        """
        result = {
            "code": error_enum.code,
            "message": error_enum.message,
            "data": None
        }
        if detail:
            result["detail"] = detail
        return result