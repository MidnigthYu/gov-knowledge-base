from abc import ABC, abstractmethod
from typing import List, Dict

class BaseReranker(ABC):
    """重排序抽象基类"""

    @abstractmethod
    def rerank(self, query: str, documents: List[Dict], top_n: int = 3) -> List[Dict]:
        """对文档列表做语义重排序"""
        pass