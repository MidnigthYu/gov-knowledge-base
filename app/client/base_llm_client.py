"""
大模型客户端统一抽象基类模块
封装单轮/多轮对话、流式生成、历史截断等通用能力，供智谱、DeepSeek 等具体客户端继承复用
依赖：requests、LLMAPIError、app.config.settings
"""
from abc import ABC, abstractmethod
import requests, json
from app.common.logger import get_logger
from app.common.exceptions import LLMAPIError
from app.config.settings import settings

logger = get_logger(__name__)

class BaseLLMClient(ABC):
    """大模型客户端统一抽象基类，支持单轮/多轮对话、流式生成与历史截断"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = None,
        max_history_rounds: int = 10,
        system_prompt: str = None
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout if timeout is not None else settings.LLM_REQUEST_TIMEOUT
        self.max_history_rounds = max_history_rounds  # 最大保留历史轮数，避免上下文过长
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        self.messages = []

        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def chat(self, prompt: str) -> str:
        """单轮/多轮对话入口，自带历史记忆，连续调用即构成多轮对话

        Args:
            prompt: 用户输入文本，必须为非空字符串

        Returns:
            大模型生成的回答文本

        Raises:
            ValueError: prompt 为空或非字符串时抛出
            LLMAPIError: 请求超时、HTTP 错误或未知异常时统一封装抛出
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
        """清空对话历史开启新会话，仅保留首个系统提示词（基础人设）"""
        system_msg = next((m for m in self.messages if m["role"] == "system"), None)
        self.messages = [system_msg] if system_msg else []
        logger.info("对话历史已清空")

    def complete(self, messages: list[dict]) -> str:
        """无状态单次补全：使用传入的完整消息列表请求模型，不读写 self.messages

        用于问题改写等不依赖会话、且不应污染主对话历史的内部调用

        Args:
            messages: 完整的消息列表，格式为 [{"role": ..., "content": ...}, ...]

        Returns:
            模型补全返回的文本

        Raises:
            LLMAPIError: 请求超时、HTTP 错误或未知异常时统一封装抛出
        """
        payload = self._build_payload(messages)
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return self._parse_response(response.json())
        except requests.exceptions.Timeout:
            logger.error(f"大模型调用超时，模型: {self.model}")
            raise LLMAPIError(f"模型 {self.model} 请求超时", status_code=408)
        except requests.exceptions.HTTPError as e:
            logger.error(f"大模型HTTP错误: {str(e)}, 状态码: {response.status_code}")
            raise LLMAPIError(f"接口调用失败: {str(e)}", status_code=response.status_code)
        except Exception as e:
            logger.error(f"大模型调用未知错误: {str(e)}")
            raise LLMAPIError(f"调用失败: {str(e)}")

    def chat_stream(self, prompt: str):
        """流式对话入口：逐块返回增量文本，自带历史记忆

        Args:
            prompt: 用户输入文本，必须为非空字符串

        Yields:
            str: 逐块生成的增量文本

        Raises:
            ValueError: prompt 为空或非字符串时抛出
            LLMAPIError: 请求超时、HTTP 错误或未知异常时统一封装抛出
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt 必须为非空字符串")

        self.messages.append({"role": "user", "content": prompt})
        self._truncate_history()

        payload = self._build_payload(self.messages)
        payload["stream"] = True
        full_answer = ""

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()

            # 逐行解析 SSE 流式响应，兼容格式差异
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                if not line.startswith("data:"):
                    continue
                
                data_str = line[5:].strip()
                
                if data_str == "[DONE]":
                    break

                try:
                    chunk_data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                delta = self._parse_stream_chunk(chunk_data)
                if delta:
                    full_answer += delta
                    yield delta

            # 完整回答存入历史，保持多轮对话能力
            self.messages.append({"role": "assistant", "content": full_answer})

        except requests.exceptions.Timeout:
            logger.error(f"大模型流式调用超时，模型：{self.model}")
            raise LLMAPIError(f"模型 {self.model} 请求超时", status_code=408)
        except requests.exceptions.HTTPError as e:
            logger.error(f"大模型流式HTTP错误：{str(e)}，状态码：{response.status_code}")
            raise LLMAPIError(f"接口调用失败：{str(e)}", status_code=response.status_code)
        except Exception as e:
            logger.error(f"大模型流式调用未知错误：{str(e)}")
            raise LLMAPIError(f"调用失败：{str(e)}")
         
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

    @abstractmethod
    def _parse_stream_chunk(self, chunk_data: dict) -> str | None:
        """解析单条流式响应块，返回增量文本；结束块返回 None"""
        pass