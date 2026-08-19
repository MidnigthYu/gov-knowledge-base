import pytest
from common.response import ResponseUtil
from common.exceptions import ErrorCode


def test_success_structure():
    """成功响应结构符合统一规范"""
    data = {"id": 1, "name": "测试"}
    resp = ResponseUtil.success(data)
    assert resp["code"] == 0
    assert resp["data"] == data
    assert "message" in resp


def test_error_structure():
    """错误响应错误码与信息一致"""
    resp = ResponseUtil.error(ErrorCode.PARAM_INVALID)
    assert resp["code"] == ErrorCode.PARAM_INVALID.code
    assert resp["message"] == ErrorCode.PARAM_INVALID.message
    assert resp["data"] is None


def test_error_with_detail():
    """异常详情字段正确透传"""
    detail = "question字段不能为空"
    resp = ResponseUtil.error(ErrorCode.PARAM_INVALID, detail=detail)
    assert resp["detail"] == detail