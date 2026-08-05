from client.base_llm_client import BaseLLMClient
from config.settings import settings

class ZhipuClient(BaseLLMClient):
    def __init__(self):
        super().__init__(
            api_key=settings.ZHIPU_API_KEY,
            base_url=settings.ZHIPU_BASE_URL,
            model="glm-4-flash",
            system_prompt="你是一个政务政策咨询助手，回答要严谨准确。"
        )

    def _build_payload(self, messages: list[dict]) -> dict:
        return {
            "model": self.model,
            "messages": messages  # 直接把历史列表传进去
        }

    def _parse_response(self, response_data: dict) -> str:
        return response_data["choices"][0]["message"]["content"]