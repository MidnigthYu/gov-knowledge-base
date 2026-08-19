import requests
from typing import List
from client.base_embedding_client import BaseEmbeddingClient
from config.settings import settings
from common.logger import get_logger

logger = get_logger(__name__)

class ZhipuEmbeddingClient(BaseEmbeddingClient):
    """智谱AI嵌入模型客户端"""

    def __init__(self):
        super().__init__(dimension=settings.EMBEDDING_DIMENSION)
        self.api_key = settings.ZHIPU_API_KEY
        self.base_url = settings.ZHIPU_BASE_URL.rstrip("/")
        self.model = settings.ZHIPU_EMBEDDING_MODEL
        self.batch_max = settings.EMBEDDING_BATCH_MAX
        self.timeout = settings.LLM_REQUEST_TIMEOUT

    def embed_single(self, text: str) -> list[float]:
        """单文本向量化，走基类校验后调用底层实现"""
        return self._embed_single_impl(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量文本向量化，走基类校验后调用底层实现"""
        return self._embed_batch_impl(texts)

    def _embed_single_impl(self, text: str) -> List[float]:
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {"model": self.model, "input": text}
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]

    def _embed_batch_impl(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        all_vectors = []
        # 自动分批
        for i in range(0, len(texts), self.batch_max):
            batch = texts[i:i + self.batch_max]
            payload = {"model": self.model, "input": batch}
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            # 按index排序保证顺序一致
            batch_vectors = sorted(data["data"], key=lambda x: x["index"])
            all_vectors.extend([item["embedding"] for item in batch_vectors])
        return all_vectors