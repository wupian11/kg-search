"""
LLM客户端

支持 OpenAI 和 GLM(智谱AI) 接口
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


class GLMClient(LLMClient):
    """
    智谱AI GLM客户端

    使用 OpenAI 兼容接口调用智谱AI服务
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
    ):
        """
        初始化GLM客户端

        Args:
            api_key: 智谱AI API密钥
            api_base: API基础URL
            model: 模型名称
        """
        settings = get_settings()

        self.api_key = api_key or settings.glm_api_key
        self.api_base = api_base or settings.glm_api_base
        self.model = model or settings.glm_model
        self.max_retries = settings.glm_max_retries
        self.timeout = settings.glm_timeout

        if not self.api_key:
            raise ValueError("GLM API key is required")

        # 智谱AI 兼容 OpenAI 接口
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout,
        )

        logger.info("GLM client initialized", model=self.model, base_url=self.api_base)

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
            temperature: 温度参数 (GLM范围 0-1)
            max_tokens: 最大token数
            response_format: 响应格式

        Returns:
            生成的回复文本
        """
        # GLM温度范围是0-1，确保在范围内
        temperature = max(0.01, min(1.0, temperature))

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        # GLM 支持 JSON 模式
        if response_format and response_format.get("type") == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = await self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""

            logger.debug(
                "GLM chat completion successful",
                model=self.model,
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
            )

            return content

        except Exception as e:
            logger.error("GLM chat completion failed", error=str(e))
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
        temperature = max(0.01, min(1.0, temperature))

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
            logger.error("GLM stream completion failed", error=str(e))
            raise

    async def close(self) -> None:
        """关闭客户端"""
        await self.client.close()


def create_llm_client(provider: str | None = None) -> LLMClient:
    """
    工厂函数：根据配置创建LLM客户端

    Args:
        provider: 服务提供商，可选 "openai" 或 "glm"，默认从配置读取

    Returns:
        LLM客户端实例
    """
    settings = get_settings()
    provider = provider or settings.llm_provider

    if provider == "glm":
        return GLMClient()
    elif provider == "openai":
        return OpenAIClient()
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
