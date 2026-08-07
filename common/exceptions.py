# common/exceptions.py
class GovRAGBaseError(Exception):
    """项目基础异常类，所有业务异常的父类"""
    pass

class LLMError(GovRAGBaseError):
    """大模型相关异常基类"""
    pass

class LLMAPIError(GovRAGBaseError):
    """大模型接口调用异常"""
    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code

class ConfigError(GovRAGBaseError):
    """配置项异常"""
    pass

class FileProcessError(GovRAGBaseError):
    """文件处理异常"""
    pass

class VectorStoreError(GovRAGBaseError):
    """向量数据库操作异常"""
    pass

class EmbeddingError(GovRAGBaseError):
    """嵌入模型调用与处理异常"""
    pass