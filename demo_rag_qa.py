from client.zhipu_client import ZhipuClient
from client.zhipu_embedding_client import ZhipuEmbeddingClient
from vector_store.chroma_store import ChromaVectorStore
from service.rag_service import RagService
from service.knowledge_manager import KnowledgeManager
from common.logger import get_logger

logger = get_logger(__name__)


def main():
    print("=" * 60)
    print("  政务政策知识库 RAG 问答系统演示")
    print("=" * 60)

    # 初始化核心组件
    print("\n[1/4] 正在初始化核心组件...")
    embedding_client = ZhipuEmbeddingClient()
    llm_client = ZhipuClient()
    vector_store = ChromaVectorStore(
        collection_name="gov_policy_demo",
        persist=False
    )
    print("✅ 组件初始化完成")

    # 批量构建知识库
    print("\n[2/4] 正在从目录构建知识库...")
    import os
    docs_dir = "./data/docs"

    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"⚠️  已自动创建目录 {docs_dir}")
        print("请放入政务政策 TXT 文档后重新运行脚本")
        return

    kb_manager = KnowledgeManager(embedding_client, vector_store)
    build_result = kb_manager.build_from_dir(docs_dir)

    print(f"✅ 知识库构建完成")
    print(f"   总文件数：{build_result['total_files']}")
    print(f"   成功处理：{build_result['success_files']}")
    print(f"   总分块数：{build_result['total_chunks']}")

    if build_result["failed_files"] > 0:
        print(f"⚠️  失败文件 {build_result['failed_files']} 个：")
        for detail in build_result["failed_details"]:
            print(f"     - {detail}")

    # 初始化 RAG 问答服务
    print("\n[3/4] 正在初始化问答服务...")
    rag_service = RagService(
        vector_store=vector_store,
        llm_client=llm_client,
        embedding_client=embedding_client
    )
    print("✅ 问答服务就绪")

    # 交互式问答循环
    print("\n[4/4] 进入问答模式，输入 exit 退出")
    print("-" * 60)

    while True:
        user_input = input("\n请输入您的问题：").strip()
        if user_input.lower() in ["exit", "quit", "退出"]:
            print("\n感谢使用，再见！")
            break
        if not user_input:
            continue

        try:
            result = rag_service.query(user_input)
            print("\n💡 回答：")
            print(result["answer"])
            print(f"\n📌 命中参考片段：{result['hit_count']} 条")
            for i, src in enumerate(result["sources"]):
                source_file = src.get("metadata", {}).get("source_file", "未知来源")
                preview = src["content"][:60].replace("\n", " ")
                print(f"   片段{i+1} [{source_file}]：{preview}...")
        except Exception as e:
            print(f"❌ 问答失败：{str(e)}")


if __name__ == "__main__":
    main()