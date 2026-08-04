import requests
from config.settings import settings
from common.logger import get_logger

logger = get_logger("zhipu_client")

class ZhipuClient:
    """智谱AI大模型客户端"""

    def __init__(self):
        self.api_key = settings.ZHIPU_API_KEY
        self.base_url = settings.ZHIPU_BASE_URL
        self.model = settings.DEFAULT_ZHIPU_MODEL

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
            logger.error(f"智谱AI调用失败: {str(e)}")
            return ""