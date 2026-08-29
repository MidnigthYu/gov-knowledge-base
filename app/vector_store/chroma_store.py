"""
Chroma 向量数据库实现模块
基于 chromadb 提供持久化/内存两种模式的向量存储，封装文档入库、相似度检索、集合管理能力
依赖：chromadb、BaseVectorStore、VectorStoreError、app.config.settings
"""
import uuid
import chromadb
from typing import List, Dict
from chromadb.config import Settings as ChromaSettings
from app.vector_store.base_vector_store import BaseVectorStore
from app.common.exceptions import VectorStoreError
from app.common.logger import get_logger
from app.config.settings import settings

logger = get_logger(__name__)

class ChromaVectorStore(BaseVectorStore):
    """Chroma 向量数据库实现，支持持久化/内存模式与多集合管理"""

    def __init__(
        self,
        collection_name: str = None,
        persist: bool = True,
        persist_dir: str = None
    ):
        """初始化 Chroma 客户端与目标集合

        Args:
            collection_name: 集合名称，缺省取配置 CHROMA_DEFAULT_COLLECTION
            persist: 是否持久化，True 使用 PersistentClient，False 使用 EphemeralClient
            persist_dir: 持久化目录，缺省取配置 CHROMA_PERSIST_DIR

        Raises:
            VectorStoreError: Chroma 初始化失败时抛出
        """
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
                metadata={"hnsw:space": "cosine", "app": "gov-rag"}
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
        collection_name: str = None
    ) -> None:
        """批量插入文档向量到目标集合

        Args:
            texts: 文档文本列表
            embeddings: 与文本对应的向量列表
            metadatas: 文档元数据列表，缺省自动生成空元数据
            ids: 文档唯一标识列表，缺省自动生成 UUID
            collection_name: 目标集合名称，缺省使用默认集合

        Raises:
            VectorStoreError: 文本/向量为空、数量不一致或入库失败时抛出
        """
        if not texts or not embeddings:
            raise VectorStoreError("文本和向量不能为空")
        if len(texts) != len(embeddings):
            raise VectorStoreError("文本数量与向量数量不一致")

        target_name = collection_name if collection_name else self.collection_name
        target_collection = self.client.get_or_create_collection(target_name, metadata={"hnsw:space": "cosine", "app": "gov-rag"})


        try:
            doc_ids = ids or [str(uuid.uuid4()) for _ in texts]

            if metadatas is None:
                metadatas = [{} for _ in texts]
            enhanced_metadatas = []
            for idx, meta in enumerate(metadatas):
                enhanced_meta = meta.copy()
                if not enhanced_meta:
                    enhanced_meta["chunk_index"] = idx
                enhanced_metadatas.append(enhanced_meta)

            target_collection.add(
                ids=doc_ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=enhanced_metadatas
            )
            logger.info(f"成功入库 {len(texts)} 条文档向量")
        except Exception as e:
            logger.error(f"向量入库失败: {str(e)}")
            raise VectorStoreError(f"文档入库失败: {str(e)}") from e

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        collection_name: str = None,
        similarity_threshold: float = 0.0
    ) -> List[Dict]:
        """相似度检索，返回按相似度降序排列的结果

        Args:
            query_embedding: 查询向量
            top_k: 期望返回的片段数量，会按集合实际文档数自动截断
            collection_name: 目标集合名称，缺省使用默认集合
            similarity_threshold: 相似度过滤阈值，低于该值的结果被丢弃

        Returns:
            含 content、metadata、similarity 字段的结果列表

        Raises:
            VectorStoreError: 查询向量为空、集合不存在或检索失败时抛出
        """
        if not query_embedding:
            raise VectorStoreError("查询向量不能为空")

        # 确定目标集合名称
        target_collection_name = collection_name if collection_name else self.collection_name

        # 严格校验集合存在性，不存在直接抛出业务异常
        if not self.collection_exists(target_collection_name):
            raise VectorStoreError(f"集合{target_collection_name}不存在")

        if target_collection_name != self.collection_name:
            target_collection = self.client.get_collection(target_collection_name)
        else:
            target_collection = self.collection

        # 文档数量做边界校验
        top_k = min(top_k, target_collection.count())
        if top_k <= 0:
            return []

        try:
            result = target_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            # 格式化返回
            items = []
            for doc, meta, dist in zip(
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0]
            ):
                similarity = round(1 - dist, 4)
                if similarity < similarity_threshold:
                    continue
                items.append({
                    "content": doc,
                    "metadata": meta,
                    "similarity": similarity
                })
                
            # 强制按相似度倒序排列
            items.sort(key=lambda x: x["similarity"], reverse=True)
            logger.info(f"向量检索：原始召回{len(result['documents'][0])}条，阈值过滤后剩余{len(items)}条，阈值={similarity_threshold}")

            return items
            
        except Exception as e:
            logger.error(f"向量检索失败: {str(e)}")
            raise VectorStoreError(f"检索失败: {str(e)}") from e

    def collection_exists(self, collection_name: str) -> bool:
        """按业务标记判断集合是否真实存在，过滤系统集合与脏集合

        Args:
            collection_name: 待判断的集合名称

        Returns:
            集合存在且带 gov-rag 业务标记时为 True，否则 False
        """
        try:
            col = self.client.get_collection(collection_name)
            return col.metadata and col.metadata.get("app") == "gov-rag"
        except Exception:
            return False

    def delete_collection(self, collection_name: str = None) -> None:
        """删除知识库集合，不传参则删除当前默认集合

        Args:
            collection_name: 待删除集合名称，缺省使用默认集合

        Raises:
            VectorStoreError: 删除集合失败时抛出
        """
        target = collection_name or self.collection_name
        try:
            self.client.delete_collection(target)
            logger.info(f"集合 {target} 已删除")
        except Exception as e:
            logger.error(f"删除集合失败: {str(e)}")
            raise VectorStoreError(f"删除集合失败: {str(e)}") from e

    def list_collections(self) -> list[str]:
        """获取所有业务知识库集合名称，过滤 Chroma 系统集合

        Returns:
            带 gov-rag 业务标记的集合名称列表
        """
        all_cols = self.client.list_collections()
        return [col.name for col in all_cols if col.metadata and col.metadata.get("app") == "gov-rag"]

    def count(self) -> int:
        """查询默认集合内文档数量

        Returns:
            集合内文档数量

        Raises:
            VectorStoreError: 统计失败时抛出
        """
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"统计文档数失败: {str(e)}")
            raise VectorStoreError(f"统计失败: {str(e)}") from e

    def delete_by_ids(self, ids: list[str], collection_name: str = None) -> None:
        """根据片段ID列表删除指定向量

        Args:
            ids: 待删除的向量片段ID列表
            collection_name: 目标知识库集合，不传则使用实例默认集合

        Raises:
            VectorStoreError: 集合不存在或删除失败时抛出
        """

        if not ids:
            logger.debug("删除向量ID列表为空，跳过操作")
            return

        target_collection = collection_name or self.collection_name

        if not self.collection_exists(target_collection):
            raise VectorStoreError(f"知识库集合不存在：{target_collection}")

        try:
            if target_collection != self.collection_name:
                collection = self.client.get_collection(name=target_collection)
            else:
                collection = self.collection

            collection.delete(ids=ids)
            logger.info(f"集合[{target_collection}] 成功删除向量片段 {len(ids)} 条")

        except Exception as e:
            logger.error(f"删除向量片段失败，集合[{target_collection}]，错误：{str(e)}")
            raise VectorStoreError(f"删除向量片段失败：{str(e)}") from e
