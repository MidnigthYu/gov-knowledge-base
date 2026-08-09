import os
from common.logger import get_logger
from common.exceptions import EmbeddingError, VectorStoreError, KnowledgeNotFoundException, DocumentParseException
from utils.text_cleaner import safe_read_file, clean_single_text
from utils.text_splitter import TextSplitter

logger = get_logger(__name__)

class KnowledgeManager:
    """
    知识库批量构建管理器
    支持目录级文档一键入库，单文件失败不中断整体任务，自动完成清洗、分块、向量化、入库全流程
    """

    def __init__(self, embedding_client, vector_store, chunk_size: int = 500, chunk_overlap: int = 50):
        """依赖注入初始化"""
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.default_collection = vector_store.collection_name

    def build_from_dir(self, input_dir: str) -> dict:
        """从指定目录递归扫描所有 TXT 文件，批量构建知识库"""
        # 递归扫描目录下所有 txt 文件
        txt_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(".txt"):
                    txt_files.append(os.path.join(root, file))

        if not txt_files:
            logger.warning(f"目录 {input_dir} 下未找到任何 txt 文件")
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
                raw_text = safe_read_file(file_path)
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
                embeddings = self.embedding_client.batch_embed(all_texts)
                self.vector_store.add_documents(all_texts, embeddings, all_metadatas)
                logger.info(f"批量入库完成，共 {total_chunks} 个文档分块")
            except EmbeddingError as e:
                logger.error(f"批量向量化失败：{str(e)}")
                raise
            except VectorStoreError as e:
                logger.error(f"向量库批量入库失败：{str(e)}")
                raise

        # 返回统计结果
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
        """获取所有知识库集合名称列表"""
        return self.vector_store.list_collections()

    def delete_knowledge_base(self, collection_name: str) -> None:
        """删除指定知识库集合"""
        all_collections = self.list_knowledge_bases()
        if collection_name not in all_collections:
            raise KnowledgeNotFoundException(f"知识库「{collection_name}」不存在")

        self.vector_store.delete_collection(collection_name)
        logger.info(f"知识库「{collection_name}」删除成功")

    def add_single_document(self, file_path: str, collection_name: str = None) -> int:
        """单文档增量入库，处理逻辑与批量构建完全对齐"""
        target_collection = collection_name or self.default_collection

        try:
            # 文档读取 + 文本清洗
            raw_text = safe_read_file(file_path)
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

        except Exception as e:
            raise DocumentParseException(f"文档解析失败: {str(e)}") from e

        try:
            # 批量向量化
            embeddings = self.embedding_client.batch_embed(all_texts)

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