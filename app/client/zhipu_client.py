"""
智谱 AI 大模型客户端模块
继承 BaseLLMClient，完成智谱特有的请求体构造与响应/流式解析，并注入政务咨询系统提示词
依赖：BaseLLMClient、app.config.settings
"""
from app.client.base_llm_client import BaseLLMClient
from app.config.settings import settings

class ZhipuClient(BaseLLMClient):
    """智谱 AI 大模型客户端，绑定 glm-4-flash 模型与政务咨询人设"""

    def __init__(self):
        super().__init__(
            api_key=settings.ZHIPU_API_KEY,
            base_url=f"{settings.ZHIPU_BASE_URL.rstrip('/')}/chat/completions",
            model="glm-4-flash",
            system_prompt="你是一个政务政策咨询助手，回答要严谨准确。"
        )

    def _build_payload(self, messages: list[dict]) -> dict:
        """构造智谱请求体"""
        return {
            "model": self.model,
            "messages": messages
        }

    def _parse_response(self, response_data: dict) -> str:
        """从智谱响应体中提取回答文本"""
        return response_data["choices"][0]["message"]["content"]

    def _parse_stream_chunk(self, chunk_data: dict) -> str | None:
        """解析智谱流式响应块，返回增量文本"""
        choices = chunk_data.get("choices", [])
        if not choices:
            return None
        delta = choices[0].get("delta", {})
        return delta.get("content", "")