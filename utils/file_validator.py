from pathlib import Path

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}

DEFAULT_MAX_SIZE = 10 * 1024 * 1024

def validate_file(filename: str, file_size: int, max_size: int = DEFAULT_MAX_SIZE) -> bool:
    """校验文件格式与大小"""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"不支持的文件格式，仅支持：{', '.join(ALLOWED_EXTENSIONS)}")

    if file_size > max_size:
        raise ValueError(f"文件大小超限，最大允许 {max_size // 1024 // 1024}MB")

    return True