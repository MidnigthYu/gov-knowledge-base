"""
智谱 AI 嵌入模型客户端模块
继承 BaseEmbeddingClient，实现智谱嵌入接口的单条/批量向量化，支持自动分批与结果顺序对齐
依赖：requests、BaseEmbeddingClient、app.config.settings
"""
import requests
from typing import List
from app.client.base_embedding_client import BaseEmbeddingClient
from app.config.settings import settings
from app.common.logger import get_logger

logger = get_logger(__name__)

class ZhipuEmbeddingClient(BaseEmbeddingClient):
    """智谱 AI 嵌入模型客户端，封装嵌入接口调用与批量处理"""

    def __init__(self):
        super().__init__(dimension=settings.EMBEDDING_DIMENSION)
        self.api_key = settings.ZHIPU_API_KEY
        self.base_url = settings.ZHIPU_BASE_URL.strip().strip('"').strip("'").rstrip("/")
        self.model = settings.ZHIPU_EMBEDDING_MODEL
        self.batch_max = settings.EMBEDDING_BATCH_MAX
        self.timeout = settings.LLM_REQUEST_TIMEOUT

    def embed(self, text: str) -> list[float]:
        """统一嵌入接口，兼容上层业务调用"""
        return self.embed_single(text)

    def embed_query(self, text: str) -> list[float]:
        """单条文本向量化，复用批量接口实现"""
        return self.embed_batch([text])[0]

    def embed_single(self, text: str) -> list[float]:
        """单文本向量化，走基类校验后调用底层实现"""
        return self._embed_single_impl(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量文本向量化，走基类校验后调用底层实现"""
        return self._embed_batch_impl(texts)

    def _embed_single_impl(self, text: str) -> List[float]:
        """调用智谱嵌入接口，将单条文本转为向量"""
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
        """按批次调用智谱嵌入接口，并按返回 index 排序保证与输入顺序一致"""
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        all_vectors = []
        # 自动分批，避免单次请求超过模型输入上限
        for i in range(0, len(texts), self.batch_max):
            batch = texts[i:i + self.batch_max]
            payload = {"model": self.model, "input": batch}
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            # 按 index 排序保证返回顺序与输入一致
            batch_vectors = sorted(data["data"], key=lambda x: x["index"])
            all_vectors.extend([item["embedding"] for item in batch_vectors])
        return all_vectors