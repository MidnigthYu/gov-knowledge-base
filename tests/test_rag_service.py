import unittest
from unittest.mock import MagicMock
from service.rag_service import RagService
from common.exceptions import LLMAPIError, VectorStoreError, EmbeddingError

class TestRagService(unittest.TestCase):
    """RAG 问答服务单元测试"""

    def setUp(self):
        # Mock 所有底层依赖，不调用真实 API
        self.mock_vector_store = MagicMock()
        self.mock_llm_client = MagicMock()
        self.mock_embedding = MagicMock()
        self.mock_reranker = MagicMock()

        # 初始化 RAG 服务，传入完整依赖
        self.rag_service = RagService(
            vector_store=self.mock_vector_store,
            llm_client=self.mock_llm_client,
            embedding_client=self.mock_embedding,
            top_k=2
        )
        self.rag_service.reranker = self.mock_reranker

    def test_empty_search_return_default_answer(self):
        """检索无结果时，返回兜底提示，且不调用大模型"""
        # 模拟向量化成功
        self.mock_embedding.embed.return_value = [0.1] * 1024
        # 模拟检索返回空
        self.mock_vector_store.search.return_value = []

        result = self.rag_service.query("测试问题")
        self.assertEqual(result["hit_count"], 0)
        self.assertEqual(len(result["sources"]), 0)
        self.assertIn("暂无", result["answer"])
        # 验证未调用大模型
        self.mock_llm_client.chat.assert_not_called()

    def test_normal_query_return_structured_result(self):
        """正常检索时，返回结构化答案与来源片段"""
        # 模拟向量化成功
        self.mock_embedding.embed.return_value = [0.1] * 1024
        # 模拟检索结果
        self.mock_vector_store.search.return_value = [
            {"content": "测试片段1", "metadata": {"source_file": "test.txt"}, "similarity": 0.9},
            {"content": "测试片段2", "metadata": {"source_file": "test.txt"}, "similarity": 0.85}
        ]
        # 模拟大模型返回
        self.mock_llm_client.chat.return_value = "这是基于参考内容的答案"

        result = self.rag_service.query("测试问题", enable_rerank=False)
        self.assertEqual(result["answer"], "这是基于参考内容的答案")
        self.assertEqual(result["hit_count"], 2)
        self.assertEqual(len(result["sources"]), 2)
        # 验证大模型被调用一次
        self.mock_llm_client.chat.assert_called_once()
        # 验证调用参数名与实现一致
        self.mock_llm_client.chat.assert_called_with(prompt="测试问题")

    def test_vector_store_error_raise_exception(self):
        """向量库异常时，封装抛出 VectorStoreError"""
        self.mock_embedding.embed.return_value = [0.1] * 1024
        self.mock_vector_store.search.side_effect = Exception("数据库连接失败")

        with self.assertRaises(VectorStoreError):
            self.rag_service.query("测试问题")

    def test_llm_error_raise_exception(self):
        """大模型调用异常时，封装抛出 LLMAPIError"""
        self.mock_embedding.embed.return_value = [0.1] * 1024
        self.mock_vector_store.search.return_value = [
            {"content": "片段1", "metadata": {"source_file": "test.txt"}, "similarity": 0.9}
        ]
        self.mock_llm_client.chat.side_effect = Exception("接口调用超时")

        with self.assertRaises(LLMAPIError):
            self.rag_service.query("测试问题", enable_rerank=False)

    def test_rerank_disabled_not_call_reranker(self):
        """关闭重排序时，不调用重排序客户端"""
        self.mock_embedding.embed.return_value = [0.1] * 1024
        self.mock_vector_store.search.return_value = [
            {"content": "测试片段", "metadata": {"source_file": "test.txt"}, "similarity": 0.9}
        ]
        self.mock_llm_client.chat.return_value = "测试答案"

        self.rag_service.query("测试问题", enable_rerank=False)

        # 验证重排序客户端未被调用
        self.mock_reranker.rerank.assert_not_called()

    def test_rerank_enabled_call_reranker_with_correct_params(self):
        """开启重排序时，正常调用客户端并使用重排序结果"""
        self.mock_embedding.embed.return_value = [0.1] * 1024
        # 模拟初筛2条结果
        self.mock_vector_store.search.return_value = [
            {"content": "片段A", "metadata": {"source_file": "a.txt"}, "similarity": 0.7},
            {"content": "片段B", "metadata": {"source_file": "b.txt"}, "similarity": 0.6}
        ]
        # 模拟重排序后返回1条最优结果
        self.mock_reranker.rerank.return_value = [
            {"content": "片段B", "metadata": {"source_file": "b.txt"}, "similarity": 0.92}
        ]
        self.mock_llm_client.chat.return_value = "基于重排序结果的答案"

        result = self.rag_service.query("测试问题", enable_rerank=True, rerank_top_n=1)

        # 验证重排序被调用1次，参数正确
        self.mock_reranker.rerank.assert_called_once()
        call_kwargs = self.mock_reranker.rerank.call_args.kwargs
        self.assertEqual(call_kwargs["top_n"], 1)
        # 验证最终结果使用重排序后的数量
        self.assertEqual(result["hit_count"], 1)
        self.assertEqual(len(result["sources"]), 1)


if __name__ == "__main__":
    unittest.main()