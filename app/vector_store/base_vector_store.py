"""
向量存储抽象基类模块
定义文档入库、相似度检索、集合删除与计数等统一接口契约，供 Chroma 等具体实现继承
依赖：Python 标准库 abc/typing、get_logger
"""
from abc import ABC, abstractmethod
from typing import List, Dict
from app.common.logger import get_logger

logger = get_logger(__name__)

class BaseVectorStore(ABC):
    """向量存储抽象基类，定义向量库操作统一接口"""

    @abstractmethod
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict] = None,
        ids: List[str] = None
    ) -> None:
        """批量插入文档向量

        Args:
            texts: 文档文本列表
            embeddings: 与文本对应的向量列表
            metadatas: 文档元数据列表，缺省为 None
            ids: 文档唯一标识列表，缺省为 None 由实现自动生成
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict]:
        """相似度检索，返回按相似度降序排列的结果

        Args:
            query_embedding: 查询向量
            top_k: 返回片段数量

        Returns:
            含 content、metadata、similarity 字段的结果列表
        """
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """删除当前集合"""
        pass

    @abstractmethod
    def count(self) -> int:
        """查询集合内文档数量

        Returns:
            集合内文档数量
        """
        pass

    @abstractmethod
    def delete_by_ids(
        self, 
        ids: List[str], 
        collection_name: str = None
    ) -> None:
        """根据片段ID列表删除指定向量

        Args:
            ids: 待删除的向量片段ID列表
            collection_name: 目标知识库集合，不传则使用实例默认集合
        """
        pass
