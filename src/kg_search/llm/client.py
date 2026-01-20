"""
OpenAI LLM客户端

封装OpenAI API调用
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from kg_search.config import get_settings
from kg_search.utils import get_logger

logger = get_logger(__name__)


class LLMClient(ABC):
    """LLM客户端抽象基类"""

    @abstractmethod
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """生成聊天回复"""
        pass

    @abstractmethod
    async def chat_completion_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """流式生成聊天回复"""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API客户端"""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
    ):
        """
        初始化OpenAI客户端

        Args:
            api_key: API密钥
            api_base: API基础URL
            model: 模型名称
        """
        settings = get_settings()

        self.api_key = api_key or settings.openai_api_key
        self.api_base = api_base or settings.openai_api_base
        self.model = model or settings.openai_model
        self.max_retries = settings.openai_max_retries
        self.timeout = settings.openai_timeout

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base if self.api_base != "https://api.openai.com/v1" else None,
            timeout=self.timeout,
        )

        logger.info("OpenAI client initialized", model=self.model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """
        生成聊天回复

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式 (如 {"type": "json_object"})

        Returns:
            生成的回复文本
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = await self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""

            logger.debug(
                "Chat completion successful",
                model=self.model,
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
            )

            return content

        except Exception as e:
            logger.error("Chat completion failed", error=str(e))
            raise

    async def chat_completion_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """
        流式生成聊天回复

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数

        Yields:
            生成的文本片段
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }

        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        try:
            stream = await self.client.chat.completions.create(**kwargs)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error("Stream completion failed", error=str(e))
            raise

    async def close(self) -> None:
        """关闭客户端"""
        await self.client.close()
