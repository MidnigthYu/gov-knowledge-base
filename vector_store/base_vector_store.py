from abc import ABC, abstractmethod
from typing import List, Dict
from common.logger import get_logger

logger = get_logger(__name__)

class BaseVectorStore(ABC):
    """向量存储抽象基类"""

    @abstractmethod
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict] = None,
        ids: List[str] = None
    ) -> None:
        """批量插入文档向量"""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict]:
        """相似度检索，返回按相似度降序排列的结果"""
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """删除当前集合"""
        pass

    @abstractmethod
    def count(self) -> int:
        """查询集合内文档数量"""
        pass