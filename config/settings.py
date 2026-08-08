import os
from dotenv import load_dotenv
from common.exceptions import ConfigError

class Settings:
    """项目统一配置管理类"""

    def __init__(self, env_file: str = ".env"):

        load_dotenv(env_file, override=False)

        # 大模型配置
        self.ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
        self.ZHIPU_BASE_URL = os.getenv("ZHIPU_BASE_URL", 
        "https://open.bigmodel.cn/api/paas/v4")

        self.DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
        self.DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", 
        "https://api.deepseek.com")

        self.LLM_REQUEST_TIMEOUT: int = int(os.getenv("LLM_REQUEST_TIMEOUT", "30"))

        # 日志配置
        self.LOG_DIR = os.getenv("LOG_DIR", "logs")
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

        # 嵌入模型配置
        self.ZHIPU_EMBEDDING_MODEL: str = os.getenv( "ZHIPU_EMBEDDING_MODEL", "embedding-3")
        self.EMBEDDING_DIMENSION: int = int (os.getenv("EMBEDDING_DIMENSION", "2048"))
        self.EMBEDDING_BATCH_MAX: int = int (os.getenv("EMBEDDING_BATCH_MAX", "256"))

        # 向量库配置
        self.CHROMA_PERSIST_DIR: str = os.getenv( "CHROMA_PERSIST_DIR","./data/chroma")
        self.CHROMA_DEFAULT_COLLECTION: str = os.getenv("CHROMA_DEFAULT_COLLECTION","gov_policy_base")
        self.VECTOR_DEFAULT_TOP_K: int = int (os.getenv("VECTOR_DEFAULT_TOP_K", "5"))

        # RAG配置
        self.RAG_DEFAULT_TOP_K: int = int (os.getenv("RAG_DEFAULT_TOP_K", "3"))

        # API 服务配置
        self.SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
        self.SERVER_PORT: int = int(os.getenv("SERVER_PORT", 8000))
        
        # 启动时执行必填项校验
        self._validate_required()

    def _validate_required(self):
        """校验必填配置项，缺失立即报错"""
        missing = []
        if not self.ZHIPU_API_KEY:
            missing.append("ZHIPU_API_KEY")
        if not self.DEEPSEEK_API_KEY:
            missing.append("DEEPSEEK_API_KEY")

        if missing:
            raise ConfigError(f"缺失必填配置项: {', '.join(missing)}，请检查 .env 文件")

# 全局单例
settings = Settings()