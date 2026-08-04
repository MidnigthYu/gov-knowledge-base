import requests
from config.settings import settings
from common.logger import get_logger

logger = get_logger("deepseek_client")

class DeepSeekClient:
    """DeepSeek大模型客户端"""

    def __init__(self):
        self.api_key = settings.DEEPSEEK_API_KEY
        self.base_url = settings.DEEPSEEK_BASE_URL
        self.model = settings.DEFAULT_DEEPSEEK_MODEL

    def chat(self, prompt: str) -> str:
        """单轮对话调用"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"DeepSeek调用失败: {str(e)}")
            return ""