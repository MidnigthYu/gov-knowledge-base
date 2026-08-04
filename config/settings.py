import os
from dotenv import load_dotenv

# 加载根目录的 .env 文件
load_dotenv()


class Settings:
    """项目全局配置"""
    # 智谱AI配置
    ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
    ZHIPU_BASE_URL = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions")

    # DeepSeek配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions")

    # 默认模型名称
    DEFAULT_ZHIPU_MODEL = "glm-4-flash"
    DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"


# 全局单例，直接导入使用
settings = Settings()