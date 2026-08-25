"""
日志工具模块
提供统一的日志器工厂，首次调用时创建日志目录并装配控制台（UTF-8）与按日滚动文件双输出
依赖：logging 标准库、app.config.settings
"""
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

def get_logger(name: str) -> logging.Logger:
    """获取配置好的日志器，首次调用时创建日志目录并装配处理器"""
    from app.config.settings import settings

    logger = logging.getLogger(name)
    if logger.handlers:  # 避免重复添加处理器
        return logger

    os.makedirs(settings.LOG_DIR, exist_ok=True)

    logger.setLevel(settings.LOG_LEVEL)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 控制台输出强制 UTF-8 编码
    stdout_utf8 = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False)
    console_handler = logging.StreamHandler(stream=stdout_utf8)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件滚动输出
    file_handler = TimedRotatingFileHandler(
        os.path.join(settings.LOG_DIR, "app.log"),
        when="midnight",
        backupCount=7,
        encoding="utf-8",
        delay=True
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger