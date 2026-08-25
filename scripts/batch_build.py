"""
批量文档入库命令行工具
递归扫描指定目录下的 TXT 文档，支持增量更新与基于 MD5 指纹的重复文档跳过，执行清洗、分块、向量化、入库全流程
依赖：KnowledgeManager、ZhipuEmbeddingClient、ChromaVectorStore、app.config.settings
"""
import os
import sys
import argparse
import hashlib
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config.settings import settings
from app.client.zhipu_embedding_client import ZhipuEmbeddingClient
from app.vector_store.chroma_store import ChromaVectorStore
from app.service.knowledge_manager import KnowledgeManager
from app.common.logger import get_logger

logger = get_logger(__name__)

def calculate_file_md5(file_path: str) -> str:
    """计算文件MD5指纹，基于二进制内容生成唯一标识"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def init_kb_manager(collection_name: str = None) -> tuple[KnowledgeManager, ChromaVectorStore]:
    """初始化知识库管理器与向量库实例，复用项目统一配置"""
    target_collection = collection_name or settings.CHROMA_DEFAULT_COLLECTION
    embedding_client = ZhipuEmbeddingClient()
    vector_store = ChromaVectorStore(
        collection_name=target_collection,
        persist_dir=settings.CHROMA_PERSIST_DIR
    )
    kb_manager = KnowledgeManager(
        embedding_client=embedding_client,
        vector_store=vector_store
    )
    return kb_manager, vector_store


def get_existing_file_md5s(vector_store: ChromaVectorStore) -> set:
    """查询目标集合中所有已入库文档的MD5，用于前置去重"""
    try:
        result = vector_store.collection.get(include=["metadatas"])
        md5_set = set()
        for meta in result["metadatas"]:
            if meta and "file_md5" in meta:
                md5_set.add(meta["file_md5"])
        return md5_set
    except Exception as e:
        logger.warning(f"获取已有文档指纹失败，将执行全量入库: {e}")
        return set()


def scan_txt_files(dir_path: str) -> list[str]:
    """递归扫描目录下所有 .txt 文档，与现有KnowledgeManager能力对齐"""
    file_list = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(".txt"):
                file_list.append(os.path.join(root, file))
    return file_list


def main():
    parser = argparse.ArgumentParser(description="政务 RAG 批量文档入库工具")
    parser.add_argument(
        "--dir",
        required=True,
        help="待入库文档所在目录路径，支持递归扫描子目录"
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="目标知识库集合名称，不传则使用环境配置中的默认集合"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制全量重建，跳过去重校验，重新入库所有文档"
    )
    args = parser.parse_args()

    target_collection = args.collection or settings.CHROMA_DEFAULT_COLLECTION
    abs_dir = os.path.abspath(args.dir)

    print("=" * 50)
    print(f"批量入库启动")
    print(f"文档目录: {abs_dir}")
    print(f"目标集合: {target_collection}")
    print(f"强制重建: {'开启' if args.force else '关闭'}")
    print("=" * 50)

    start_time = time.time()
    total_files = 0
    added_count = 0
    skipped_count = 0
    failed_count = 0

    try:
        kb_manager, vector_store = init_kb_manager(args.collection)

        file_list = scan_txt_files(abs_dir)
        total_files = len(file_list)
        print(f"\n共扫描到有效文档: {total_files} 个")

        # 非强制模式下，加载已有文档指纹
        existing_md5s = set()
        if not args.force:
            existing_md5s = get_existing_file_md5s(vector_store)
            print(f"集合中已有文档指纹数: {len(existing_md5s)}")

        # 逐文件判重、入库
        for file_path in file_list:
            file_md5 = calculate_file_md5(file_path)

            if not args.force and file_md5 in existing_md5s:
                skipped_count += 1
                print(f"[跳过] 文档已存在: {os.path.basename(file_path)}")
                continue

            try:
                chunk_num = kb_manager.add_single_document(
                    file_path=file_path,
                    collection_name=target_collection,
                    extra_metadata={"file_md5": file_md5}
                )
                added_count += 1
                print(f"[入库成功] {os.path.basename(file_path)} | 分块数: {chunk_num}")
            except Exception as e:
                failed_count += 1
                logger.error(f"文档入库失败 {file_path}: {e}")
                print(f"[入库失败] {os.path.basename(file_path)}: {str(e)}")

        cost_time = round(time.time() - start_time, 2)
        print("\n" + "=" * 50)
        print(f"入库执行完成，总耗时: {cost_time}s")
        print(f"总扫描文件数: {total_files}")
        print(f"新增入库: {added_count}")
        print(f"跳过重复: {skipped_count}")
        print(f"入库失败: {failed_count}")
        print("=" * 50)

    except Exception as e:
        print(f"\n入库执行异常: {str(e)}")
        logger.error(f"批量入库失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
