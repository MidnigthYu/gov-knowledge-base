from client.base_llm_client import BaseLLMClient
from config.settings import settings

class DeepSeekClient(BaseLLMClient):
    def __init__(self):
        super().__init__(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=f"{settings.DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions",
            model="deepseek-chat"
        )

    def _build_payload(self, messages: list[dict]) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7
        }

    def _parse_response(self, response_data: dict) -> str:
        return response_data["choices"][0]["message"]["content"]

    def _parse_stream_chunk(self, chunk_data: dict) -> str | None:
        """解析DeepSeek流式响应块，返回增量文本"""
        choices = chunk_data.get("choices", [])
        if not choices:
            return None
        delta = choices[0].get("delta", {})
        return delta.get("content", "")