"""
业务异常体系模块
定义统一错误码枚举与分层异常类，覆盖参数、文件、大模型、嵌入、向量库、问答等全部业务异常场景
依赖：Python 标准库 enum
"""
from enum import Enum

class ErrorCode(Enum):
    """统一业务错误码枚举，值结构为 (错误码, 错误描述)"""
    # 参数配置类
    PARAM_INVALID = (1001, "参数校验失败")
    CONFIG_ERROR = (1002, "配置项异常")
    NOT_FOUND = (1003, "请求的资源不存在")

    # 知识库文档类
    KNOWLEDGE_NOT_FOUND = (2001, "知识库不存在")
    DOCUMENT_PARSE_FAILED = (2002, "文档解析失败")

    # 文件文档类
    FILE_FORMAT_NOT_SUPPORTED = (3101, "文件格式不支持，仅支持PDF/Word/TXT")
    FILE_SIZE_EXCEED = (3102, "文件大小超出限制，最大支持10MB")

    # 大模型与嵌入类
    LLM_API_ERROR = (3001, "大模型接口调用异常")
    EMBEDDING_ERROR = (3002, "嵌入模型调用异常")
    RERANK_ERROR = (3003, "重排序服务调用异常")

    # 向量库类
    VECTOR_STORE_ERROR = (4001, "向量数据库操作异常")

    # 问答业务类
    RETRIEVAL_EMPTY = (4101, "未检索到相关文档片段")

    # 系统通用类
    SYSTEM_ERROR = (5000, "系统内部错误")
    
    @property
    def code(self) -> int:
        """返回错误码数值部分"""
        return self.value[0]

    @property
    def message(self) -> str:
        """返回错误描述文本部分"""
        return self.value[1]


class GovRAGBaseError(Exception):
    """项目基础异常类，所有业务异常的父类"""
    default_error_code = ErrorCode.SYSTEM_ERROR

    def __init__(self, error_code_or_message=None, detail: str = None):
        # 传枚举对象 / 传消息字符串
        if isinstance(error_code_or_message, ErrorCode):
            # 传入枚举对象 + 可选详情
            self.error_code = error_code_or_message
            self.detail = detail
            self.message = self.error_code.message
            if detail:
                self.message = f"{self.message}：{detail}"
        else:
            # 传入消息字符串，错误码取子类默认值
            self.error_code = self.default_error_code
            self.detail = None
            self.message = error_code_or_message or self.error_code.message

        self.code = self.error_code.code
        super().__init__(self.message)


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


class ParamInvalidError(GovRAGBaseError):
    """参数校验不合法异常"""
    default_error_code = ErrorCode.PARAM_INVALID
    pass


class KnowledgeNotFoundError(GovRAGBaseError):
    """知识库集合不存在异常"""
    default_error_code = ErrorCode.KNOWLEDGE_NOT_FOUND
    pass


class DocumentParseError(FileProcessError):
    """文档解析失败异常，复用文件处理错误码"""
    default_error_code = ErrorCode.DOCUMENT_PARSE_FAILED
    pass


class RerankError(GovRAGBaseError):
    """重排序服务调用异常"""
    default_error_code = ErrorCode.RERANK_ERROR
    pass

