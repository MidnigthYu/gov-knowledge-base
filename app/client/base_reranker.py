"""
重排序客户端统一抽象基类模块
定义语义重排序的统一接口契约，供智谱等具体重排序客户端继承实现
依赖：Python 标准库 abc/typing
"""
from abc import ABC, abstractmethod
from typing import List, Dict

class BaseReranker(ABC):
    """重排序抽象基类，定义语义精排的统一接口"""

    @abstractmethod
    def rerank(self, query: str, documents: List[Dict], top_n: int = 3) -> List[Dict]:
        """对候选文档列表做语义重排序

        Args:
            query: 查询文本
            documents: 待重排序的候选文档列表，每个元素含 content/metadata 字段
            top_n: 精排后返回的片段数量

        Returns:
            按相关性降序排列的文档列表
        """
        pass