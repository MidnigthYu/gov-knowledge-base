"""
嵌入模型客户端统一抽象基类模块
封装单条/批量向量化的入参校验、维度校验与异常封装，供具体嵌入客户端继承实现
依赖：EmbeddingError、get_logger
"""
from abc import ABC, abstractmethod
from typing import List
from app.common.exceptions import EmbeddingError
from app.common.logger import get_logger

logger = get_logger(__name__)

class BaseEmbeddingClient(ABC):
    """嵌入模型客户端抽象基类，统一向量化入参校验与异常封装"""

    def __init__(self, dimension: int):
        self.dimension = dimension

    def embed_single(self, text: str) -> List[float]:
        """单文本向量化（统一入参校验 + 维度校验 + 异常封装）

        Args:
            text: 待向量化文本

        Returns:
            文本对应的向量

        Raises:
            EmbeddingError: 文本为空、维度不匹配或底层调用失败时抛出
        """
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
        """批量文本向量化（统一校验 + 分批处理）

        Args:
            texts: 待向量化文本列表

        Returns:
            与输入顺序一致的向量列表

        Raises:
            EmbeddingError: 入参非法、全部为空或底层调用失败时抛出
        """
        if not texts or not all(isinstance(t, str) for t in texts):
            raise EmbeddingError("批量向量化入参必须为非空字符串列表")
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
    def _embed_batch_impl(self, texts: List[str]) -> List[List[float]]:
        """子类实现：批量向量化具体逻辑"""
        pass