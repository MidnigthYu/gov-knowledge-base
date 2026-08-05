from abc import ABC, abstractmethod
import requests
from common.logger import get_logger
from common.exceptions import LLMAPIError

logger = get_logger(__name__)

class BaseLLMClient(ABC):
    """大模型客户端统一抽象基类，支持单轮/多轮对话"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = 30,
        max_history_rounds: int = 10,
        system_prompt: str = None
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_history_rounds = max_history_rounds  # 最大保留历史轮数，避免上下文过长
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        self.messages = []

        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def chat(self, prompt: str) -> str:
        """
        对话入口：自带历史记忆，连续调用就是多轮对话
        调用 clear_history() 可开启新会话
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt 必须为非空字符串")

        self.messages.append({"role": "user", "content": prompt})

        self._truncate_history()

        payload = self._build_payload(self.messages)

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            answer = self._parse_response(response.json())

            self.messages.append({"role": "assistant", "content": answer})
            return answer

        except requests.exceptions.Timeout:
            logger.error(f"大模型调用超时，模型: {self.model}")
            raise LLMAPIError(f"模型 {self.model} 请求超时", status_code=408)
        except requests.exceptions.HTTPError as e:
            logger.error(f"大模型HTTP错误: {str(e)}, 状态码: {response.status_code}")
            raise LLMAPIError(f"接口调用失败: {str(e)}", status_code=response.status_code)
        except Exception as e:
            logger.error(f"大模型调用未知错误: {str(e)}")
            raise LLMAPIError(f"调用失败: {str(e)}")

    def clear_history(self):
        """清空对话历史，开启新会话"""
        system_msg = [m for m in self.messages if m["role"] == "system"]
        self.messages = system_msg.copy()
        logger.info("对话历史已清空")

    def _truncate_history(self):
        """
        内部方法：截断历史消息，控制上下文长度
        保留系统提示词 + 最近的对话轮次
        """
        system_msg = [m for m in self.messages if m["role"] == "system"]
        dialog_msg = [m for m in self.messages if m["role"] != "system"]

        max_messages = self.max_history_rounds * 2
        if len(dialog_msg) > max_messages:
            dialog_msg = dialog_msg[-max_messages:]

        self.messages = system_msg + dialog_msg

    @abstractmethod
    def _build_payload(self, messages: list[dict]) -> dict:
        """构造请求体，子类必须实现；入参改为完整消息列表"""
        pass

    @abstractmethod
    def _parse_response(self, response_data: dict) -> str:
        """解析响应结果，子类必须实现"""
        pass