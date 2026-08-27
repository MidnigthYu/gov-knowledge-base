"""
知识库管理服务模块
提供目录级批量构建、单文档增量入库、列表查询与删除等能力，串联清洗、分块、向量化、入库全流程
依赖：extract_text、clean_single_text、TextSplitter、EmbeddingError、VectorStoreError 等
"""
import os
from app.common.logger import get_logger
from app.common.exceptions import EmbeddingError, VectorStoreError, KnowledgeNotFoundError, DocumentParseError
from app.utils.text_cleaner import clean_single_text
from app.utils.text_splitter import TextSplitter
from app.utils.format_parser import extract_text

logger = get_logger(__name__)


class KnowledgeManager:
    """
    知识库批量构建管理器
    支持目录级文档一键入库，单文件失败不中断整体任务，自动完成清洗、分块、向量化、入库全流程
    支持格式：TXT、Markdown、PDF、DOCX（文本型）
    """
    def __init__(self, embedding_client, vector_store, chunk_size: int = 500, chunk_overlap: int = 50):
        """依赖注入初始化
        Args:
            embedding_client: 嵌入客户端实例，负责文本向量化
            vector_store: 向量存储实例，负责文档入库与检索
            chunk_size: 分块长度，缺省 500
            chunk_overlap: 分块重叠区间，缺省 50
        """
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.default_collection = vector_store.collection_name

    def build_from_dir(self, input_dir: str, collection_name: str = None) -> dict:
        """从指定目录递归扫描所有支持格式文档，批量构建知识库
        Args:
            input_dir: 待入库文档目录路径
            collection_name: 目标知识库集合名称，缺省使用默认集合
        Returns:
            含 total_files、success_files、failed_files、total_chunks、failed_details 的统计字典
        Raises:
            DocumentParseError: 目录不存在或读取失败时抛出
            EmbeddingError: 批量向量化失败时抛出
            VectorStoreError: 向量库批量入库失败时抛出
        """
        # 先校验目录合法性
        if not os.path.isdir(input_dir):
            logger.warning(f"目录不存在 {input_dir}")
            raise DocumentParseError(f"目录不存在 / 读取失败：{input_dir}")
        
        # 递归扫描目录下所有支持格式文档
        txt_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith((".txt", ".md", ".pdf", ".docx")):
                    txt_files.append(os.path.join(root, file))

        if not txt_files:
            logger.warning(f"目录 {input_dir} 下未找到任何支持格式的文档")
            return {
                "total_files": 0,
                "success_files": 0,
                "failed_files": 0,
                "total_chunks": 0,
                "failed_details": []
            }

        all_texts = []
        all_metadatas = []
        success_count = 0
        failed_count = 0
        failed_details = []

        for file_path in txt_files:
            try:
                raw_text = extract_text(file_path)
                if not raw_text:
                    logger.warning(f"文档解析失败，跳过：{file_path}")
                    failed_count += 1
                    failed_details.append({"file": file_path, "reason": "格式解析失败"})
                    continue

                clean_text = clean_single_text(raw_text)
                if not clean_text.strip():
                    logger.info(f"文件 {file_path} 内容为空，跳过入库")
                    success_count += 1
                    continue

                chunks = self.splitter.split(clean_text)
                if not chunks:
                    success_count += 1
                    continue

                file_rel_path = os.path.relpath(file_path, input_dir)
                for chunk in chunks:
                    all_texts.append(chunk["content"])
                    all_metadatas.append({
                        "source_file": file_rel_path,
                        "chunk_index": chunk["chunk_index"],
                        "start_pos": chunk["start_pos"]
                    })
                success_count += 1
                logger.info(f"文件 {file_path} 处理完成，生成 {len(chunks)} 个分块")

            except Exception as e:
                # 异常隔离：单个文件失败仅记录，不中断整体流程
                failed_count += 1
                err_msg = f"{file_path}: {str(e)}"
                failed_details.append(err_msg)
                logger.error(f"文件处理失败：{err_msg}")

        # 批量向量化 + 批量入库
        total_chunks = len(all_texts)
        if total_chunks > 0:
            try:
                embeddings = self.embedding_client.embed_batch(all_texts)
                self.vector_store.add_documents(
                    texts = all_texts, 
                    embeddings = embeddings, 
                    metadatas = all_metadatas,
                    collection_name = collection_name
                )
                logger.info(f"批量入库完成，共 {total_chunks} 个文档分块")
            except EmbeddingError as e:
                logger.error(f"批量向量化失败：{str(e)}")
                raise
            except VectorStoreError as e:
                logger.error(f"向量库批量入库失败：{str(e)}")
                raise

        result = {
            "total_files": len(txt_files),
            "success_files": success_count,
            "failed_files": failed_count,
            "total_chunks": total_chunks,
            "failed_details": failed_details
        }
        logger.info(f"知识库构建任务结束：{result}")
        return result

    def list_knowledge_bases(self) -> list[str]:
        """获取所有知识库集合名称列表
        Returns:
            业务知识库集合名称列表
        """
        return self.vector_store.list_collections()

    def delete_knowledge_base(self, collection_name: str) -> None:
        """删除指定知识库集合
        Args:
            collection_name: 待删除的知识库集合名称
        Raises:
            KnowledgeNotFoundError: 集合不存在时抛出
        """
        if not self.vector_store.collection_exists(collection_name):
            raise KnowledgeNotFoundError(f"知识库 [{collection_name}] 不存在")
        self.vector_store.delete_collection(collection_name)
        logger.info(f"知识库 [{collection_name}] 删除成功")

    def add_single_document(self, file_path: str, collection_name: str = None, extra_metadata: dict = None) -> int:
        """单文档增量入库，处理逻辑与批量构建完全对齐
        支持格式：TXT、Markdown、PDF、DOCX（文本型）
        Args:
            file_path: 待入库文档的本地路径
            collection_name: 目标知识库集合名称，缺省使用默认集合
            extra_metadata: 额外元数据，会合并到每个分块元数据中
        Returns:
            实际入库的分块数量
        Raises:
            DocumentParseError: 文档解析失败时抛出
            EmbeddingError: 批量向量化失败时抛出
            VectorStoreError: 向量库入库失败时抛出
        """
        target_collection = collection_name or self.default_collection
        try:
            # 接入统一格式解析，和批量构建逻辑对齐
            raw_text = extract_text(file_path)
            if not raw_text:
                raise DocumentParseError(f"文档格式不支持或解析失败：{file_path}")

            clean_text = clean_single_text(raw_text)
            # 空内容前置校验
            if not clean_text.strip():
                logger.warning(f"文档 {file_path} 内容为空，跳过入库")
                return 0

            # 文本分块
            chunk_dicts = self.splitter.split(clean_text)
            if not chunk_dicts:
                logger.warning(f"文档 {file_path} 分块后无有效内容，跳过入库")
                return 0

            # 提取纯文本与元数据，格式与批量构建完全统一
            all_texts = []
            all_metadatas = []
            for chunk in chunk_dicts:
                chunk_content = chunk.get("content", "")
                if not chunk_content.strip():
                    continue
                all_texts.append(chunk_content)
                all_metadatas.append({
                    "source_file": file_path,
                    "chunk_index": chunk.get("chunk_index", 0),
                    "start_pos": chunk.get("start_pos", 0)
                })

            if not all_texts:
                return 0

            if extra_metadata:
                for meta in all_metadatas:
                    meta.update(extra_metadata)

        except Exception as e:
            raise DocumentParseError(f"文档解析失败: {str(e)}") from e

        try:
            # 批量向量化
            embeddings = self.embedding_client.embed_batch(all_texts)
            # 写入目标向量库
            self.vector_store.add_documents(
                texts=all_texts,
                embeddings=embeddings,
                metadatas=all_metadatas,
                collection_name=target_collection
            )
        except Exception as e:
            logger.error(f"文档入库失败: {str(e)}")
            raise

        logger.info(f"文档 {file_path} 入库完成，新增 {len(all_texts)} 个分块")
        return len(all_texts)
