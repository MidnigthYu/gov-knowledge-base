"""
智谱重排序客户端单元测试
以 Mock 网络请求，验证空输入、正常解析、异常封装与缺元数据安全处理
依赖：unittest、ZhipuReranker
"""
import unittest
from unittest.mock import patch, MagicMock
from app.client.zhipu_reranker import ZhipuReranker
from app.common.exceptions import RerankError

class TestZhipuReranker(unittest.TestCase):
    """智谱重排序客户端单元测试"""

    def setUp(self):
        self.reranker = ZhipuReranker()
        self.test_docs = [
            {"content": "测试文本1", "metadata": {"source": "a.txt"}, "similarity": 0.7},
            {"content": "测试文本2", "metadata": {"source": "b.txt"}, "similarity": 0.8}
        ]

    def test_empty_documents_return_empty_list(self):
        """空输入时，直接返回空列表，不发起网络请求"""
        result = self.reranker.rerank("测试问题", [])
        self.assertEqual(result, [])

    @patch("app.client.zhipu_reranker.requests.post")
    def test_normal_rerank_return_structured_result(self, mock_post):
        """正常调用时，正确解析返回结果并还原元数据"""
        # 模拟接口返回
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.88}
            ]
        }
        mock_post.return_value = mock_response

        result = self.reranker.rerank("测试问题", self.test_docs, top_n=2)

        # 验证返回数量与顺序
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["content"], "测试文本2")
        self.assertEqual(result[0]["metadata"]["source"], "b.txt")
        self.assertEqual(result[0]["similarity"], 0.95)
        self.assertEqual(result[1]["content"], "测试文本1")

        # 验证请求参数正确
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args.kwargs["json"]["model"], self.reranker.model)
        self.assertEqual(call_args.kwargs["json"]["top_n"], 2)

    @patch("app.client.zhipu_reranker.requests.post")
    def test_api_error_raise_rerank_error(self, mock_post):
        """接口调用异常时，封装抛出 RerankError"""
        # 模拟网络异常
        mock_post.side_effect = Exception("连接超时")

        with self.assertRaises(RerankError):
            self.reranker.rerank("测试问题", self.test_docs)

    @patch("app.client.zhipu_reranker.requests.post")
    def test_missing_metadata_safe_handle(self, mock_post):
        """输入缺少 metadata 字段时，安全处理不报错"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.9}]
        }
        mock_post.return_value = mock_response

        # 输入无 metadata
        docs_without_meta = [{"content": "测试片段"}]
        result = self.reranker.rerank("测试问题", docs_without_meta)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["metadata"], {})


if __name__ == "__main__":
    unittest.main()