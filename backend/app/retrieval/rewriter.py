"""查询改写：指代消解 + 多版本扩展

- 异步、失败自动降级为原查询（不会因为 LLM 抽风而 500）；
- 只在「有对话历史」或「短查询」时触发，避免每次提问都多花一次调用。
"""
from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from ..core.config import config
from ..core.llm import get_client

logger = logging.getLogger(__name__)


class QueryRewriter:
    PROMPT = """你是检索查询改写器。根据对话历史，把用户问题改写成 1-3 条独立、适合检索的查询。

要求：
1. 出现「它 / 这个 / 那个 / 上面说的」等指代词时，替换成历史里明确的实体；
2. 补全省略的信息，但不要添加原问题没有的假设；
3. 保持原意，输出简短陈述句；
4. 只输出 JSON：{{"queries": ["改写1", "改写2"]}}

对话历史：
{history}

用户问题：{query}"""

    async def rewrite(
        self,
        query: str,
        history: Optional[List[dict]] = None,
        use_llm: bool = False,
    ) -> List[str]:
        if not use_llm or not config.llm.configured:
            return [query]

        try:
            client = get_client()
            response = await client.chat.completions.create(
                model=config.llm.model,
                messages=[
                    {
                        "role": "user",
                        "content": self.PROMPT.format(
                            history=self._format_history(history or []) or "无",
                            query=query,
                        ),
                    }
                ],
                temperature=0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            payload = self._parse_json(response.choices[0].message.content or "")
            queries = [str(q).strip() for q in payload.get("queries", []) if str(q).strip()]
        except Exception as exc:
            logger.warning("查询改写失败，使用原查询：%s", exc)
            return [query]

        result: List[str] = []
        for item in [query, *queries]:
            if item not in result:
                result.append(item)
        return result[:3]

    @staticmethod
    def _format_history(history: List[dict]) -> str:
        lines: List[str] = []
        for turn in history[-config.history_turns:]:
            question = str(turn.get("question", "")).strip()
            answer = str(turn.get("answer", "")).strip()
            if question:
                lines.append(f"用户：{question}")
            if answer:
                lines.append(f"助手：{answer[:200]}")
        return "\n".join(lines)

    @staticmethod
    def _parse_json(text: str) -> dict:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {}
