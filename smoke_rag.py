from utils.text_cleaner import clean_single_text
from utils.text_splitter import TextSplitter
from client import ZhipuEmbeddingClient
from vector_store import ChromaVectorStore

def main():
    # 1. 模拟政务文档
    raw_text = """
    武汉市惠企纾困政策清单
    一、稳岗返还政策
    参保企业上年度未裁员或裁员率不高于5.5%，30人以下参保企业裁员率不高于参保职工总数20%的，可以申请失业保险稳岗返还。
    大型企业返还比例为企业及其职工上年度实际缴纳失业保险费的30%，中小微企业返还比例为60%。

    二、一次性留工培训补助
    2023年1月1日至12月31日，累计出现1个以上中高风险疫情地区的区，可对因疫情严重影响暂时无法正常生产经营的中小微企业，按每名参保职工500元的标准发放一次性留工培训补助。

    三、创业担保贷款
    符合条件的个人创业者可申请最高20万元创业担保贷款，期限最长不超过3年，财政部门给予全额贴息。
    小微企业当年新招用符合条件人员达到现有在职职工人数15%以上的，可申请最高300万元创业担保贷款。
    """

    # 2. 文本清洗
    clean_text = clean_single_text(raw_text)
    print(f"清洗后文本长度: {len(clean_text)}")

    # 3. 文本分块
    splitter = TextSplitter(chunk_size=200, chunk_overlap=30)
    chunks = splitter.split(clean_text)
    print(f"分块数量: {len(chunks)}")
    chunk_texts = [c["content"] for c in chunks]

    # 4. 批量向量化
    embed_client = ZhipuEmbeddingClient()
    vectors = embed_client.embed_batch(chunk_texts)
    print(f"生成向量数量: {len(vectors)}")

    # 5. 向量入库（内存模式）
    store = ChromaVectorStore(collection_name="smoke_test", persist=False)
    metadatas = [{"chunk_index": c["chunk_index"]} for c in chunks]
    store.add_documents(chunk_texts, vectors, metadatas=metadatas)
    print(f"向量库文档总数: {store.count()}")

    # 6. 检索测试
    query = "创业贷款最多能贷多少钱"
    query_vec = embed_client.embed_single(query)
    results = store.search(query_vec, top_k=3)

    print("\n=== 检索结果 ===")
    for idx, r in enumerate(results):
        print(f"Top{idx+1} 相似度:{r['similarity']}")
        print(f"内容: {r['content'][:80]}...\n")


if __name__ == "__main__":
    main()