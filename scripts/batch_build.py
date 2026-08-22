import os
import sys
import argparse

# 项目根目录加入 Python 路径，解决脚本独立运行的导入问题
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from client.zhipu_embedding_client import ZhipuEmbeddingClient
from vector_store.chroma_store import ChromaVectorStore
from service.knowledge_manager import KnowledgeManager
from common.logger import get_logger

logger = get_logger(__name__)


def init_kb_manager() -> KnowledgeManager:
    """
    完全复刻 main.py lifespan 中的组件初始化逻辑
    保证脚本与服务端使用同一套配置、同一份持久化数据
    """
    embedding_client = ZhipuEmbeddingClient()
    vector_store = ChromaVectorStore(
        collection_name=settings.CHROMA_DEFAULT_COLLECTION,
        persist_dir=settings.CHROMA_PERSIST_DIR
    )
    return KnowledgeManager(
        embedding_client=embedding_client,
        vector_store=vector_store
    )


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
    args = parser.parse_args()

    # 初始化组件
    kb_manager = init_kb_manager()

    # 确定目标集合
    target_collection = args.collection or settings.CHROMA_DEFAULT_COLLECTION
    abs_dir = os.path.abspath(args.dir)

    print("=" * 50)
    print(f"批量入库启动")
    print(f"文档目录: {abs_dir}")
    print(f"目标集合: {target_collection}")
    print("=" * 50)

    try:
        # 调用原生批量入库能力，内部已实现：目录扫描 + 文本读取 + 分块 + 向量化 + 入库 + 异常隔离
        success_count = kb_manager.build_from_dir(
            input_dir=abs_dir,
            collection_name=args.collection
        )

        print("\n" + "=" * 50)
        print(f"入库完成 | 成功处理文档数: {success_count}")
        print("=" * 50)

    except Exception as e:
        print(f"\n入库执行异常: {str(e)}")
        logger.error(f"批量入库失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
