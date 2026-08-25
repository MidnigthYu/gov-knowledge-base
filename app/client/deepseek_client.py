"""
DeepSeek 大模型客户端模块
继承 BaseLLMClient，完成 DeepSeek 特有的请求体构造与响应/流式解析
依赖：BaseLLMClient、app.config.settings
"""
from app.client.base_llm_client import BaseLLMClient
from app.config.settings import settings

class DeepSeekClient(BaseLLMClient):
    """DeepSeek 大模型客户端，绑定 deepseek-chat 模型与专属参数"""

    def __init__(self):
        super().__init__(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=f"{settings.DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions",
            model="deepseek-chat"
        )

    def _build_payload(self, messages: list[dict]) -> dict:
        """构造 DeepSeek 请求体，附加固定温度参数"""
        return {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7
        }

    def _parse_response(self, response_data: dict) -> str:
        """从 DeepSeek 响应体中提取回答文本"""
        return response_data["choices"][0]["message"]["content"]

    def _parse_stream_chunk(self, chunk_data: dict) -> str | None:
        """解析 DeepSeek 流式响应块，返回增量文本"""
        choices = chunk_data.get("choices", [])
        if not choices:
            return None
        delta = choices[0].get("delta", {})
        return delta.get("content", "")