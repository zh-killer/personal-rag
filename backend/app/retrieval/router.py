"""问题路由：meta（知识库自身问题）/ summary（总结类）/ factoid（事实检索）

旧版用「多少 / 支持 / 知识库」这类泛关键词做包含匹配，导致
「这份文档有多少页？」被当成元问题直接回文档列表。这里改成整句模式匹配。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ..core.config import config

QueryType = Literal["factoid", "summary", "meta"]


@dataclass
class RoutingDecision:
    query_type: QueryType
    top_k: int
    use_rewrite: bool
    reason: str = ""


class QueryRouter:
    """规则路由：只处理「知识库自身」这类明确意图，其余一律走检索"""

    META_PATTERNS = [
        r"有(哪些|什么|多少)(文档|文件|资料)",
        r"(文档|文件|资料)(列表|清单|目录)",
        r"知识库(里)?(有|存|包含|放)了?(什么|哪些)",
        r"支持(哪些|什么)(格式|类型|文件)",
        r"(怎么|如何)(使用|用)(这个|本)?(系统|知识库)",
        r"我(都)?(上传|存|放)了(什么|哪些)",
        r"一共(有)?(多少|几)(个|篇)(文档|文件|资料)",
    ]

    SUMMARY_KEYWORDS = [
        "总结", "概括", "归纳", "概述", "摘要", "梳理", "大纲", "要点",
        "讲了什么", "说了什么", "主要内容", "核心内容", "整体讲",
    ]

    # 需要跨页/跨文档综合的问题，用「快速」模式（多片段上下文）更合适；
    # 其余问题默认走「精准」模式（选页 + 整页精读）。
    SUMMARY_CUES = SUMMARY_KEYWORDS + [
        "对比", "比较", "区别", "差异", "联系", "综述", "整体", "全部",
        "所有文档", "哪些文档", "分别", "异同", "共同点", "演进", "体系",
    ]

    def pick_mode(self, query: str) -> str:
        """自动模式：总结/对比类走 fast，其余走 precise。"""
        text = query.strip()
        if any(cue in text for cue in self.SUMMARY_CUES):
            return "fast"
        return "precise"

    def route(self, query: str, has_history: bool = False) -> RoutingDecision:
        text = query.strip()
        lowered = text.lower()

        if len(text) <= 30 and any(re.search(pattern, text) for pattern in self.META_PATTERNS):
            return RoutingDecision("meta", 0, False, "知识库元问题")

        if any(keyword in text for keyword in self.SUMMARY_KEYWORDS) or len(text) > 120:
            return RoutingDecision(
                "summary", max(12, config.retrieval.top_k), False, "总结类问题"
            )

        needs_rewrite = has_history or len(text) <= 12
        return RoutingDecision(
            "factoid",
            config.retrieval.top_k,
            needs_rewrite,
            "多轮改写" if has_history else ("短查询扩展" if needs_rewrite else "直接检索"),
        )
