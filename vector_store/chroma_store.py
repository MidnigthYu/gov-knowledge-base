import uuid
import chromadb
from typing import List, Dict
from chromadb.config import Settings as ChromaSettings
from vector_store.base_vector_store import BaseVectorStore
from common.exceptions import VectorStoreError
from common.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

class ChromaVectorStore(BaseVectorStore):
    """Chroma 向量数据库实现"""

    def __init__(
        self,
        collection_name: str = None,
        persist: bool = True,
        persist_dir: str = None
    ):
        self.collection_name = collection_name or settings.CHROMA_DEFAULT_COLLECTION
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR

        try:
            if persist:
                self.client = chromadb.PersistentClient(
                    path=self.persist_dir,
                    settings=ChromaSettings(anonymized_telemetry=False)
                )
            else:
                self.client = chromadb.EphemeralClient(
                    settings=ChromaSettings(anonymized_telemetry=False)
                )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Chroma集合初始化成功: {self.collection_name}")
        except Exception as e:
            logger.error(f"Chroma初始化失败: {str(e)}")
            raise VectorStoreError(f"向量库初始化失败: {str(e)}") from e

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict] = None,
        ids: List[str] = None,
        collection_name: str = None  # 可选目标集合名称
    ) -> None:
        if not texts or not embeddings:
            raise VectorStoreError("文本和向量不能为空")
        if len(texts) != len(embeddings):
            raise VectorStoreError("文本数量与向量数量不一致")

        # 动态获取目标集合
        if collection_name and collection_name != self.collection_name:
            target_collection = self.client.get_or_create_collection(collection_name)
        else:
            target_collection = self.collection

        try:
            doc_ids = ids or [str(uuid.uuid4()) for _ in texts]
            meta = metadatas if metadatas else None
            target_collection.add(
                ids=doc_ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=meta
            )
            logger.info(f"成功入库 {len(texts)} 条文档向量")
        except Exception as e:
            logger.error(f"向量入库失败: {str(e)}")
            raise VectorStoreError(f"文档入库失败: {str(e)}") from e

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict]:
        if not query_embedding:
            raise VectorStoreError("查询向量不能为空")
        top_k = min(top_k, self.count())
        if top_k <= 0:
            return []

        try:
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            # 格式化返回，distance转相似度
            items = []
            for doc, meta, dist in zip(
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0]
            ):
                items.append({
                    "content": doc,
                    "metadata": meta,
                    "similarity": round(1 - dist, 4)
                })
            # 按相似度降序
            return sorted(items, key=lambda x: x["similarity"], reverse=True)
        except Exception as e:
            logger.error(f"向量检索失败: {str(e)}")
            raise VectorStoreError(f"检索失败: {str(e)}") from e

    def delete_collection(self, collection_name: str = None) -> None:
        """删除知识库集合，不传参则删除当前默认集合"""
        target = collection_name or self.collection_name
        try:
            self.client.delete_collection(target)
            logger.info(f"集合 {target} 已删除")
        except Exception as e:
            logger.error(f"删除集合失败: {str(e)}")
            raise VectorStoreError(f"删除集合失败: {str(e)}") from e

    def list_collections(self) -> list[str]:
        """获取所有知识库集合名称"""
        collections = self.client.list_collections()
        return [col.name for col in collections]

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"统计文档数失败: {str(e)}")
            raise VectorStoreError(f"统计失败: {str(e)}") from e