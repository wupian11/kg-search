"""
Embedding服务

使用OpenAI text-embedding-3生成向量
"""

from abc import ABC, abstractmethod

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from kg_search.config import get_settings
from kg_search.utils import chunk_list, get_logger

logger = get_logger(__name__)


class EmbeddingService(ABC):
    """Embedding服务抽象基类"""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """生成单个文本的向量"""
        pass

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量生成文本向量"""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """向量维度"""
        pass


class OpenAIEmbedding(EmbeddingService):
    """OpenAI Embedding服务"""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
    ):
        """
        初始化Embedding服务

        Args:
            api_key: API密钥
            api_base: API基础URL
            model: 模型名称
            dimensions: 向量维度
        """
        settings = get_settings()

        self.api_key = api_key or settings.openai_api_key
        self.api_base = api_base or settings.openai_api_base
        self.model = model or settings.openai_embedding_model
        self._dimensions = dimensions or settings.openai_embedding_dimensions

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base if self.api_base != "https://api.openai.com/v1" else None,
        )

        # 批处理配置
        self.batch_size = 100  # OpenAI最大批处理数量

        logger.info(
            "OpenAI Embedding service initialized",
            model=self.model,
            dimensions=self._dimensions,
        )

    @property
    def dimensions(self) -> int:
        """向量维度"""
        return self._dimensions

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def embed_text(self, text: str) -> list[float]:
        """
        生成单个文本的向量

        Args:
            text: 输入文本

        Returns:
            向量
        """
        if not text.strip():
            return [0.0] * self._dimensions

        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self._dimensions,
            )

            return response.data[0].embedding

        except Exception as e:
            logger.error("Embedding failed", error=str(e))
            raise

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        批量生成文本向量

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        # 分批处理
        for batch in chunk_list(texts, self.batch_size):
            # 过滤空文本
            non_empty_texts = []
            empty_indices = []

            for i, text in enumerate(batch):
                if text.strip():
                    non_empty_texts.append(text)
                else:
                    empty_indices.append(i)

            if non_empty_texts:
                batch_embeddings = await self._embed_batch(non_empty_texts)

                # 插入空向量
                result = []
                embed_idx = 0
                for i in range(len(batch)):
                    if i in empty_indices:
                        result.append([0.0] * self._dimensions)
                    else:
                        result.append(batch_embeddings[embed_idx])
                        embed_idx += 1

                all_embeddings.extend(result)
            else:
                all_embeddings.extend([[0.0] * self._dimensions] * len(batch))

        logger.info("Batch embedding completed", count=len(texts))
        return all_embeddings

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        批量嵌入（内部方法）

        Args:
            texts: 非空文本列表

        Returns:
            向量列表
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self._dimensions,
            )

            # 按索引排序（API可能不保证顺序）
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]

        except Exception as e:
            logger.error("Batch embedding failed", error=str(e), batch_size=len(texts))
            raise

    async def close(self) -> None:
        """关闭客户端"""
        await self.client.close()
