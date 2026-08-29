"""政务文档统一解析入口
复用现有格式解析能力，串联政务专项清洗与基础清洗，
向上层业务提供单文档解析与批量目录解析的统一接口。
"""
import os
from typing import List, Tuple
from app.utils.format_parser import extract_text
from app.utils.text_cleaner import clean_single_text, clean_government_text
from app.common.logger import get_logger

logger = get_logger(__name__)


class DocumentParser:
    """政务文档统一解析入口
    复用现有格式解析能力，执行「政务专项清洗 → 基础清洗」链式处理
    """

    @classmethod
    def parse(cls, file_path: str) -> str:
        """解析单个文档，返回清洗后的纯文本

        Args:
            file_path: 本地文档绝对路径

        Returns:
            清洗后的纯文本内容

        Raises:
            ValueError: 文档解析失败或格式不支持
            Exception: 解析失败时抛出
        """
        raw_text = extract_text(file_path)

        # 统一校验：覆盖格式不支持、文件不存在、编码失败、解析异常等所有场景
        if raw_text is None:
            raise ValueError(f"解析失败或格式不支持：{os.path.basename(file_path)}")

        # 链式清洗：先政务专项 → 后基础清洗
        gov_clean = clean_government_text(raw_text)
        final_text = clean_single_text(gov_clean)

        return final_text

    @classmethod
    def batch_parse(cls, dir_path: str) -> List[Tuple[str, str]]:
        """批量递归解析目录下所有支持格式的文档

        Args:
            dir_path: 文档根目录路径

        Returns:
            元组列表：(文件相对路径, 清洗后文本)
        """
        results = []
        for root, _, files in os.walk(dir_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                try:
                    text = cls.parse(file_path)
                    rel_path = os.path.relpath(file_path, dir_path)
                    results.append((rel_path, text))
                except Exception as e:
                    logger.warning(f"解析文档失败 [{file_path}]：{str(e)}")
                    continue
        return results
