"""
答案生成器

基于检索结果生成回答
"""

from typing import Any, AsyncIterator

from kg_search.utils import get_logger

from .client import LLMClient

logger = get_logger(__name__)


# 系统提示词
SYSTEM_PROMPT = """你是一个专业的文博领域智能助手，专门帮助用户了解中国文物和博物馆相关知识。

你的职责：
1. 基于检索到的知识库内容准确回答问题
2. 如果检索内容不足以回答问题，诚实说明
3. 在回答中引用来源，帮助用户了解信息出处
4. 使用专业但易懂的语言

回答风格：
- 准确：基于检索到的事实
- 专业：使用正确的文物学术语
- 清晰：结构化的回答
- 有帮助：提供相关的扩展信息"""


class AnswerGenerator:
    """答案生成器"""

    def __init__(self, llm_client: LLMClient):
        """
        初始化答案生成器

        Args:
            llm_client: LLM客户端
        """
        self.llm_client = llm_client
        self.system_prompt = SYSTEM_PROMPT

    async def generate(
        self,
        query: str,
        context: str,
        include_sources: bool = True,
        max_tokens: int | None = None,
    ) -> str:
        """
        生成答案

        Args:
            query: 用户问题
            context: 检索到的上下文
            include_sources: 是否要求引用来源
            max_tokens: 最大token数

        Returns:
            生成的答案
        """
        user_prompt = self._build_user_prompt(query, context, include_sources)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            answer = await self.llm_client.chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=max_tokens,
            )

            logger.info("Answer generated", query_length=len(query))
            return answer

        except Exception as e:
            logger.error("Answer generation failed", error=str(e))
            return "抱歉，生成答案时出现了问题，请稍后重试。"

    async def generate_stream(
        self,
        query: str,
        context: str,
        include_sources: bool = True,
    ) -> AsyncIterator[str]:
        """
        流式生成答案

        Args:
            query: 用户问题
            context: 检索到的上下文
            include_sources: 是否要求引用来源

        Yields:
            答案片段
        """
        user_prompt = self._build_user_prompt(query, context, include_sources)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            async for chunk in self.llm_client.chat_completion_stream(
                messages=messages,
                temperature=0.7,
            ):
                yield chunk

        except Exception as e:
            logger.error("Stream generation failed", error=str(e))
            yield "抱歉，生成答案时出现了问题，请稍后重试。"

    async def generate_with_citations(
        self,
        query: str,
        retrieval_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        生成带引用的答案

        Args:
            query: 用户问题
            retrieval_results: 检索结果列表

        Returns:
            包含答案和引用的字典
        """
        # 构建带编号的上下文
        context_parts = []
        sources = []

        for i, result in enumerate(retrieval_results, 1):
            context_parts.append(f"[{i}] {result['content']}")
            sources.append(
                {
                    "id": i,
                    "source_id": result.get("id", ""),
                    "content_preview": result["content"][:100] + "..."
                    if len(result["content"]) > 100
                    else result["content"],
                    "metadata": result.get("metadata", {}),
                }
            )

        context = "\n\n".join(context_parts)

        # 生成答案
        prompt = f"""基于以下编号的参考资料回答用户问题。在回答中使用[1]、[2]等标记引用来源。

## 参考资料
{context}

## 用户问题
{query}

## 要求
1. 在陈述事实时标注引用来源，如"四羊方尊是商代青铜器[1]"
2. 如果多个来源支持同一观点，可以标注多个，如[1][3]
3. 如果参考资料无法回答问题，请说明

请回答："""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            answer = await self.llm_client.chat_completion(
                messages=messages,
                temperature=0.7,
            )

            return {
                "answer": answer,
                "sources": sources,
                "query": query,
            }

        except Exception as e:
            logger.error("Citation generation failed", error=str(e))
            return {
                "answer": "抱歉，生成答案时出现了问题。",
                "sources": sources,
                "query": query,
            }

    async def refine_answer(
        self,
        query: str,
        initial_answer: str,
        additional_context: str,
    ) -> str:
        """
        优化答案

        Args:
            query: 原始问题
            initial_answer: 初始答案
            additional_context: 额外上下文

        Returns:
            优化后的答案
        """
        prompt = f"""请根据新的上下文信息优化之前的回答。

## 原始问题
{query}

## 之前的回答
{initial_answer}

## 新的上下文信息
{additional_context}

## 要求
1. 如果新信息有补充，请整合到回答中
2. 如果新信息与之前回答矛盾，以更可靠的信息为准
3. 保持回答的连贯性

请给出优化后的回答："""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            answer = await self.llm_client.chat_completion(
                messages=messages,
                temperature=0.7,
            )
            return answer

        except Exception as e:
            logger.error("Answer refinement failed", error=str(e))
            return initial_answer

    def _build_user_prompt(
        self,
        query: str,
        context: str,
        include_sources: bool,
    ) -> str:
        """构建用户提示词"""
        source_instruction = ""
        if include_sources:
            source_instruction = "\n4. 如果可能，说明信息来源"

        return f"""基于以下检索到的知识库内容回答用户问题。

## 检索到的内容
{context}

## 用户问题
{query}

## 回答要求
1. 直接回答用户的问题
2. 如果检索内容不包含相关信息，请说明"根据现有知识库信息，无法回答该问题"
3. 不要编造不在检索内容中的信息{source_instruction}

请回答："""
