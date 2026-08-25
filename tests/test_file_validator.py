"""
文件校验工具单元测试
覆盖合法/非法扩展名、大小超限、空文件与大小写扩展名等边界场景
依赖：pytest、validate_file
"""
import pytest
from app.utils.file_validator import validate_file, ALLOWED_EXTENSIONS


def test_allowed_extensions_pass():
    """合法格式校验通过"""
    assert validate_file("policy.txt", 1024) is True
    assert validate_file("government.md", 2048) is True
    assert validate_file("report.pdf", 10240) is True


def test_disallowed_extensions_raise():
    """非法格式拦截报错"""
    with pytest.raises(ValueError, match="不支持的文件格式"):
        validate_file("hack.exe", 1024)
    with pytest.raises(ValueError, match="不支持的文件格式"):
        validate_file("script.bat", 512)


def test_size_limit_raise():
    """文件大小超限拦截"""
    max_size = 10 * 1024 * 1024  # 10MB
    with pytest.raises(ValueError, match="文件大小超限"):
        validate_file("big.txt", max_size + 1)


def test_empty_file_allowed():
    """空文件边界场景兼容"""
    assert validate_file("empty.txt", 0) is True


def test_case_insensitive_ext():
    """扩展名大小写不敏感"""
    assert validate_file("DOC.PDF", 1024) is True
    assert validate_file("note.TXT", 512) is True