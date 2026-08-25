"""
文件校验工具模块
提供文件扩展名白名单与大小上限校验能力，作为文件入库前的第一道防线
依赖：Python 标准库 pathlib
"""
from pathlib import Path

# 允许入库的文件扩展名白名单
ALLOWED_EXTENSIONS = {".txt", ".md"}

# 默认最大文件大小 10MB
DEFAULT_MAX_SIZE = 10 * 1024 * 1024

def validate_file(filename: str, file_size: int, max_size: int = DEFAULT_MAX_SIZE) -> bool:
    """校验文件格式与大小

    Args:
        filename: 文件名，用于提取扩展名
        file_size: 文件大小（字节）
        max_size: 允许的最大大小，缺省 10MB

    Returns:
        校验通过返回 True

    Raises:
        ValueError: 文件格式不支持或大小超限时抛出
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"不支持的文件格式，仅支持：{', '.join(ALLOWED_EXTENSIONS)}")

    if file_size > max_size:
        raise ValueError(f"文件大小超限，最大允许 {max_size // 1024 // 1024}MB")

    return True