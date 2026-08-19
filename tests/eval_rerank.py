import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import time
from config.settings import settings
from vector_store.chroma_store import ChromaVectorStore
from client.zhipu_embedding_client import ZhipuEmbeddingClient
from client.zhipu_reranker import ZhipuReranker
from dataset.policy_eval_set import EVAL_DATASET


# 全局只初始化一次客户端，避免重复创建
_embed_client = None
_vector_store = None
_reranker = None


def get_embed_client():
    global _embed_client
    if _embed_client is None:
        # 修正：仅传 dimension，api_key 类内部自动读取
        _embed_client = ZhipuEmbeddingClient()
    return _embed_client


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        # 修正：初始化参数与单元测试完全一致，连接默认知识库集合
        _vector_store = ChromaVectorStore(
            collection_name="gov_policy_base",
            persist=True
        )
    return _vector_store


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = ZhipuReranker()
    return _reranker


def run_retrieval(query: str, top_k: int):
    """纯向量粗召回，调用方式与项目代码完全一致"""
    embed_client = get_embed_client()
    vector_store = get_vector_store()
    
    # 向量化
    query_vector = embed_client.embed_single(query)
    # 修正：search 第一个参数传向量（位置参数），不使用关键字
    results = vector_store.search(query_vector, top_k=top_k)
    return results


def run_single_eval(
    enable_rerank: bool,
    recall_top_k: int,
    rerank_top_n: int
) -> dict:
    """单组参数评测"""
    hit_top1 = 0
    hit_top3 = 0
    total_time = 0
    total = len(EVAL_DATASET)
    error_cases = []
    reranker = get_reranker() if enable_rerank else None

    for item in EVAL_DATASET:
        start = time.time()

        # 1. 向量粗召回
        search_results = run_retrieval(item["query"], top_k=recall_top_k)
        if not search_results:
            error_cases.append(item["query"])
            total_time += time.time() - start
            continue

        # 2. 重排序精排
        if enable_rerank and reranker and search_results:
            original_count = len(search_results)
            actual_rerank_n = min(rerank_top_n, original_count)
            final_results = reranker.rerank(
                query=item["query"],
                documents=search_results,
                top_n=actual_rerank_n
            )
        else:
            final_results = search_results[:rerank_top_n]

        cost_ms = (time.time() - start) * 1000
        total_time += cost_ms

        # 3. 关键词命中判断
        keyword_list = item["hit_keyword"].split()
        contents = [doc["content"] for doc in final_results]

        top1_hit = all(kw in contents[0] for kw in keyword_list)
        top3_hit = any(all(kw in c for kw in keyword_list) for c in contents[:3])

        if top1_hit:
            hit_top1 += 1
        if top3_hit:
            hit_top3 += 1
        else:
            error_cases.append(item["query"])

    return {
        "enable_rerank": enable_rerank,
        "recall_top_k": recall_top_k,
        "rerank_top_n": rerank_top_n,
        "top1_hit_rate": hit_top1 / total,
        "top3_hit_rate": hit_top3 / total,
        "avg_cost_ms": total_time / total,
        "error_cases": error_cases
    }


def print_result(res: dict):
    """格式化输出评测结果"""
    status = "开启" if res["enable_rerank"] else "关闭"
    print("=" * 60)
    print(f"测试组：重排序{status} | 粗召回{res['recall_top_k']}条 | 精排{res['rerank_top_n']}条")
    print(f"Top1 命中率：{res['top1_hit_rate']:.2%}")
    print(f"Top3 命中率：{res['top3_hit_rate']:.2%}")
    print(f"平均检索耗时：{res['avg_cost_ms']:.0f} ms")
    if res["error_cases"]:
        print(f"未命中问题：{res['error_cases']}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    # ==========关闭重排序 ==========
    print("【基线测试：关闭重排序】")
    baseline = run_single_eval(
        enable_rerank=False,
        recall_top_k=3,
        rerank_top_n=3
    )
    print_result(baseline)

    # ========== 默认参数开启重排序 ==========
    print("【基准测试：默认参数开启重排序】")
    default_rerank = run_single_eval(
        enable_rerank=True,
        recall_top_k=20,
        rerank_top_n=3
    )
    print_result(default_rerank)

    # ========== 不同粗召回量对比 ==========
    print("【参数寻优：不同粗召回量对比（固定精排3条）】")
    for k in [10, 15, 20, 30]:
        res = run_single_eval(enable_rerank=True, recall_top_k=k, rerank_top_n=3)
        print_result(res)

    # ========== 不同精排数量对比 ==========
    print("【参数寻优：不同精排数量对比（固定粗召回20条）】")
    for n in [2, 3, 5]:
        res = run_single_eval(enable_rerank=True, recall_top_k=20, rerank_top_n=n)
        print_result(res)