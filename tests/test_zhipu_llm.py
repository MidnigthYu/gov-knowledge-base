import unittest
from unittest.mock import patch, MagicMock
import requests
from app.client.zhipu_client import ZhipuClient
from app.common.exceptions import LLMAPIError


class TestZhipuClientRetry(unittest.TestCase):
    def setUp(self):
        self.client = ZhipuClient()
        self.client.max_retries = 2

    # 叠加sleep mock，跳过真实等待
    @patch("app.client.base_llm_client.time.sleep")
    @patch("app.client.base_llm_client.requests.post")
    def test_retry_on_5xx_server_error(self, mock_post, mock_sleep):
        """服务端5xx错误触发重试，达到最大重试次数后抛出LLMAPIError"""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Internal Server Error",
            response=mock_resp
        )
        mock_post.return_value = mock_resp

        with self.assertRaises(LLMAPIError):
            self.client.chat("测试问题")

        self.assertEqual(mock_post.call_count, 3)
        # 重试2次，对应2次sleep
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("app.client.base_llm_client.time.sleep")
    @patch("app.client.base_llm_client.requests.post")
    def test_no_retry_on_4xx_client_error(self, mock_post, mock_sleep):
        """客户端4xx参数错误不触发重试，直接抛出异常"""
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "400 Bad Request",
            response=mock_resp
        )
        mock_post.return_value = mock_resp

        with self.assertRaises(LLMAPIError):
            self.client.chat("测试问题")

        self.assertEqual(mock_post.call_count, 1)
        # 无重试，无sleep
        self.assertEqual(mock_sleep.call_count, 0)

    @patch("app.client.base_llm_client.time.sleep")
    @patch("app.client.base_llm_client.requests.post")
    def test_retry_success_on_second_attempt(self, mock_post, mock_sleep):
        """首次失败、第二次重试成功，正常返回结果，不超额重试"""
        error_resp = MagicMock()
        error_resp.status_code = 500
        error_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Internal Server Error",
            response=error_resp
        )

        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.raise_for_status.return_value = None
        success_resp.json.return_value = {
            "choices": [{"message": {"content": "测试回答内容"}}]
        }

        mock_post.side_effect = [error_resp, success_resp]

        result = self.client.chat("测试问题")
        self.assertEqual(result, "测试回答内容")
        self.assertEqual(mock_post.call_count, 2)
        # 1次重试对应1次sleep
        self.assertEqual(mock_sleep.call_count, 1)

    @patch("app.client.base_llm_client.time.sleep")
    @patch("app.client.base_llm_client.requests.post")
    def test_retry_on_connection_error(self, mock_post, mock_sleep):
        """requests连接异常触发重试逻辑"""
        mock_post.side_effect = requests.exceptions.ConnectionError("网络连接失败")

        with self.assertRaises(LLMAPIError):
            self.client.chat("测试问题")

        self.assertEqual(mock_post.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)
