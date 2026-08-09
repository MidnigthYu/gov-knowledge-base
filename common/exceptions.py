from enum import Enum

class ErrorCode(Enum):
    """统一业务错误码枚举"""
    # 参数配置类
    PARAM_INVALID = (1001, "参数校验失败")
    CONFIG_ERROR = (1002, "配置项异常")

    # 知识库文档类
    KNOWLEDGE_NOT_FOUND = (2001, "知识库不存在")
    DOCUMENT_PARSE_FAILED = (2002, "文档解析失败")

    # 大模型与嵌入类
    LLM_API_ERROR = (3001, "大模型接口调用异常")
    EMBEDDING_ERROR = (3002, "嵌入模型调用异常")

    # 向量库类
    VECTOR_STORE_ERROR = (4001, "向量数据库操作异常")

    # 系统通用类
    SYSTEM_ERROR = (5000, "系统内部错误")

    @property
    def code(self) -> int:
        return self.value[0]

    @property
    def message(self) -> str:
        return self.value[1]


class GovRAGBaseError(Exception):
    """项目基础异常类，所有业务异常的父类"""
    default_error_code = ErrorCode.SYSTEM_ERROR

    def __init__(self, message: str = None):
        self.code = self.default_error_code.code
        self.message = message or self.default_error_code.message
        super().__init__(self.message)


class LLMError(GovRAGBaseError):
    """大模型相关异常基类"""
    default_error_code = ErrorCode.LLM_API_ERROR
    pass


class LLMAPIError(GovRAGBaseError):
    """大模型接口调用异常"""
    default_error_code = ErrorCode.LLM_API_ERROR

    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code


class ConfigError(GovRAGBaseError):
    """配置项异常"""
    default_error_code = ErrorCode.CONFIG_ERROR
    pass


class FileProcessError(GovRAGBaseError):
    """文件处理异常"""
    default_error_code = ErrorCode.DOCUMENT_PARSE_FAILED
    pass


class VectorStoreError(GovRAGBaseError):
    """向量数据库操作异常"""
    default_error_code = ErrorCode.VECTOR_STORE_ERROR
    pass


class EmbeddingError(GovRAGBaseError):
    """嵌入模型调用与处理异常"""
    default_error_code = ErrorCode.EMBEDDING_ERROR
    pass


class ParamInvalidException(GovRAGBaseError):
    """参数校验不合法异常"""
    default_error_code = ErrorCode.PARAM_INVALID
    pass


class KnowledgeNotFoundException(GovRAGBaseError):
    """知识库集合不存在异常"""
    default_error_code = ErrorCode.KNOWLEDGE_NOT_FOUND
    pass


class DocumentParseException(FileProcessError):
    """文档解析失败异常，复用文件处理错误码"""
    default_error_code = ErrorCode.DOCUMENT_PARSE_FAILED
    pass