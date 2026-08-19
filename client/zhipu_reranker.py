import requests
from common.logger import get_logger
from common.exceptions import RerankError
from config.settings import settings
from client.base_reranker import BaseReranker
from typing import List, Dict

logger = get_logger(__name__)

class ZhipuReranker(BaseReranker):
    def __init__(self):
        self.api_key = settings.ZHIPU_API_KEY
        self.model = settings.RERANK_MODEL
        self.base_url = settings.ZHIPU_BASE_URL.rstrip("/")
        self.timeout = settings.LLM_REQUEST_TIMEOUT

    def rerank(self, query: str, documents: List[Dict], top_n: int = 3) -> List[Dict]:
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