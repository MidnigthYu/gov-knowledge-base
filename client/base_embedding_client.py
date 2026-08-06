from abc import ABC, abstractmethod
from typing import List
from common.exceptions import EmbeddingError
from common.logger import get_logger

logger = get_logger(__name__)

class BaseEmbeddingClient(ABC):
    """嵌入模型客户端抽象基类"""

    def __init__(self, dimension: int):
        self.dimension = dimension

    def embed_single(self, text: str) -> List[float]:
        """单文本向量化（统一入参校验+异常封装）"""
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError("向量化文本不能为空或非字符串")
        try:
            vector = self._embed_single_impl(text.strip())
            if len(vector) != self.dimension:
                raise EmbeddingError(
                    f"向量维度不匹配：预期{self.dimension}，实际{len(vector)}"
                )
            return vector
        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(f"单文本向量化失败: {str(e)}")
            raise EmbeddingError(f"向量化调用失败: {str(e)}") from e

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量文本向量化（统一校验+分批处理）"""
        if not texts or not all(isinstance(t, str) for t in texts):
            raise EmbeddingError("批量向量化入参必须为非空字符串列表")
        # 修复：原逻辑 t in t.strip() 错误，改为判断清洗后非空
        clean_texts = [t.strip() for t in texts if t.strip()]
        if not clean_texts:
            raise EmbeddingError("批量文本全部为空")
        try:
            return self._embed_batch_impl(clean_texts)
        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(f"批量向量化失败: {str(e)}")
            raise EmbeddingError(f"批量向量化调用失败: {str(e)}") from e

    @abstractmethod
    def _embed_single_impl(self, text: str) -> List[float]:
        """子类实现：单文本向量化具体逻辑"""
        pass

    @abstractmethod
    # 修复：参数名 rexts 拼写错误，改为 texts
    def _embed_batch_impl(self, texts: List[str]) -> List[List[float]]:
        """子类实现：批量向量化具体逻辑"""
        pass