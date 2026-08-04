import re
import chardet
from pathlib import Path
from typing import Optional
from common.logger import get_logger

logger = get_logger("text_cleaner")

# 全局配置常量
SUPPORTED_EXTENSIONS = {".txt"}
DETECT_SAMPLE_BYTES = 1024
CONFIDENCE_THRESHOLD = 0.7
ENCODING_FALLBACK_LIST = ["utf-8", "gbk", "gb2312", "utf-8-sig"]


def safe_read_file(file_path: str | Path) -> Optional[str]:
    """
    自动检测编码并安全读取文件
    采用 chardet 自动检测 + 置信度把关 + 多级编码兜底方案
    """
    file_path = Path(file_path)

    try:
        with open(file_path, "rb") as f:
            raw_sample = f.read(DETECT_SAMPLE_BYTES)
    except Exception as e:
        logger.warning(f"文件读取失败 {file_path}: {str(e)}")
        return None

    detect_result = chardet.detect(raw_sample)
    encoding = detect_result.get("encoding")
    confidence = detect_result.get("confidence", 0)

    # 置信度达标则直接使用检测结果
    if encoding and confidence >= CONFIDENCE_THRESHOLD:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            logger.debug(f"chardet检测结果解码失败，启用兜底: {file_path}")

    # 置信度不足，进入多级编码兜底
    for enc in ENCODING_FALLBACK_LIST:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue

    logger.warning(f"无法识别文件编码: {file_path}")
    return None


def clean_single_text(raw_text: str) -> str:
    """
    政务场景文本清洗
    包含去空行、首尾去空白、规范连续空格、白名单过滤特殊字符
    """
    lines = raw_text.splitlines()
    clean_lines = []

    for line in lines:
        # 先去首尾空白再判空，过滤全空格构成的假空行
        if not line.strip():
            continue
        
        line_stripped = line.strip()
        # 连续空白统一为单个空格
        line_normalized = re.sub(r"\s+", " ", line_stripped)

        # 白名单过滤：仅保留中文、英文、数字及政务公文常用标点
        line_filtered = re.sub(
            r"[^\u4e00-\u9fa5a-zA-Z0-9，。；：“”‘’（）【】！？、,.!?;:()\[\]《》\- ]",
            "",
            line_normalized
        )
        clean_lines.append(line_filtered)

    return "\n".join(clean_lines)


def batch_clean_files(input_dir: str | Path, output_dir: str | Path) -> None:
    """
    递归批量清洗目录下所有目标文件，输出保持原目录结构
    单文件异常隔离，单个失败不中断整体任务
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        logger.error(f"输入目录不存在，任务终止: {input_path}")
        return
    
    output_path.mkdir(parents=True, exist_ok=True)

    file_list = [
        f for f in input_path.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    total = len(file_list)
    logger.info(f"扫描完成，共发现 {total} 个待处理文件")

    if total == 0:
        logger.warning("未找到可处理的文件，任务结束")
        return

    success_count = 0
    fail_count = 0

    for index, file_path in enumerate(file_list, start=1):
        try:
            # 计算相对路径，保持原目录层级
            rel_path = file_path.relative_to(input_path)
            output_file = output_path / rel_path
            output_file.parent.mkdir(parents=True, exist_ok=True)

            raw_text = safe_read_file(file_path)
            if raw_text is None:
                fail_count += 1
                continue

            cleaned_text = clean_single_text(raw_text)

            # 输出统一为 UTF-8，保证下游链路格式一致
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            success_count += 1
            logger.info(f"处理完成 [{index}/{total}]: {rel_path}")

        except Exception as e:
            fail_count += 1
            logger.error(f"处理失败 [{index}/{total}]: {file_path}，原因: {str(e)}")
            continue

    logger.info(f"批量处理结束：成功 {success_count} 个，失败 {fail_count} 个")