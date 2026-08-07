import os
from common.logger import get_logger
from common.exceptions import EmbeddingError, VectorStoreError
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

                # 构造每个分块的元数据（强制非空，规避 Chroma 校验错误）
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