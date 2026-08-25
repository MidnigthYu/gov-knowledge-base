"""
智谱 AI 重排序客户端模块
继承 BaseReranker，调用智谱 rerank 接口对候选片段做语义精排，并还原原始文档元数据
依赖：requests、RerankError、app.config.settings
"""
import requests
from app.common.logger import get_logger
from app.common.exceptions import RerankError
from app.config.settings import settings
from app.client.base_reranker import BaseReranker
from typing import List, Dict

logger = get_logger(__name__)

class ZhipuReranker(BaseReranker):
    """智谱 AI 重排序客户端，封装 rerank 接口调用与结果元数据还原"""

    def __init__(self):
        self.api_key = settings.ZHIPU_API_KEY
        self.model = settings.RERANK_MODEL
        self.base_url = settings.ZHIPU_BASE_URL.rstrip("/")
        self.timeout = settings.LLM_REQUEST_TIMEOUT

    def rerank(self, query: str, documents: List[Dict], top_n: int = 3) -> List[Dict]:
        """对候选文档列表调用智谱接口做语义精排

        Args:
            query: 查询文本
            documents: 候选文档列表
            top_n: 精排返回数量

        Returns:
            按相关性降序排列的文档列表，保留原 content/metadata 并回填相似度得分

        Raises:
            RerankError: 重排序服务调用异常时抛出
        """
        if not documents:
            return []

        doc_texts = [doc["content"] for doc in documents]

        try:
            payload = {
                "model": self.model,
                "query": query,
                "documents": doc_texts,
                "top_n": top_n
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            response = requests.post(f"{self.base_url}/rerank", json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            # 还原原文档元数据
            ranked_docs = []
            for item in data["results"]:
                origin_index = item["index"]
                relevance_score = item["relevance_score"]
                origin_doc = documents[origin_index]
                ranked_docs.append({
                    "content": origin_doc["content"],
                    "metadata": origin_doc.get("metadata", {}),
                    "similarity": round(relevance_score, 4)
                })

            logger.info(f"重排序完成：初筛{len(documents)}条，精排后{len(ranked_docs)}条")
            return ranked_docs

        except Exception as e:
            logger.error(f"智谱重排序调用失败：{str(e)}")
            raise RerankError(f"重排序服务异常：{str(e)}")