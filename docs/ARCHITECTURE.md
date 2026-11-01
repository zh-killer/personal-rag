# 架构说明

## 总览

```
                    ┌─────────────────────────────┐
                    │  frontend (React + Vite)    │
                    │  Chat / Search / Upload     │
                    │  DocumentViewer             │
                    └──────────────┬──────────────┘
                                   │ HTTP / SSE
                    ┌──────────────▼──────────────┐
                    │  FastAPI (backend/app)      │
                    │  routes · SSE · 静态托管     │
                    └───┬──────────┬──────────┬───┘
                        │          │          │
          ┌─────────────▼──┐  ┌────▼─────┐  ┌─▼──────────────┐
          │ indexing       │  │ retrieval│  │ services       │
          │ parser / OCR   │  │ router   │  │ tasks          │
          │ splitter       │  │ rewrite  │  │ conversations  │
          │ embedder       │  │ retrieve │  │ usage / backup │
          │ Chroma / FTS5  │  │ rerank   │  └────────────────┘
          │ vocab / layout │  │ generate │
          └───────┬────────┘  │ precise  │
                  │           │ verify   │
                  │           └────┬─────┘
                  ▼                ▼
        ┌──────────────────────────────────┐
        │ 本机存储                          │
        │ chroma_db/  data/  fts.db         │
        │ vocab.db  conversations.db        │
        │ usage.db  cache/                  │
        └──────────────────────────────────┘
```

## 入库流程

```
上传 → 202 + task_id
        │
        ▼
  parser.parse(file)                      保留页码
        │
        ├─ PDF：逐页 get_text()
        │    └─ 页面文字过少 → OCR 该页 → 用 OCR 文本
        ├─ Markdown / TXT：按页/整篇
        └─ 图片：OCR
        │
        ▼
  splitter：500 字 / 重叠 60，优先按段落、句号切
        │
        ▼
  embedder：BAAI/bge-small-zh-v1.5（512 维，离线）
        │
        ├─ Chroma：向量 + metadata(doc_id, source, page, chunk_idx)
        ├─ FTS5：CJK 单字 + 二元组，content UNINDEXED
        ├─ vocab：词表类文档抽取结构化词条
        └─ documents.json：doc_id / 文件名 / hash / 页数 / 切块数
```

要点：

- **异步入库**：`POST /api/documents` 立即返回 202，解析/OCR/向量化在 `TaskManager` 的后台线程里跑，前端轮询 `/api/tasks/{id}`
- **稳定 doc_id**：同一文件重传时复用 doc_id，按内容 hash 判断是否需要重建，避免“删了再传索引不到”
- **增量索引**：只写新文档的向量与 FTS 行，不整库重建
- **OCR 兜底**：只对“有页面但文字层几乎为空”的页调用，避免正常 PDF 变慢

## 检索与生成

```
问题
 │
 ├─ router：meta（有多少文档/能干嘛）→ 直接回答
 ├─ lookup：词条精确命中 → 直接回答（0 次模型调用）
 │
 ├─ rewrite：多轮指代消解（“它/这个”补全）
 │
 ├─ retrieve：
 │    ├─ 稠密召回（Chroma，top 40）
 │    ├─ FTS5 召回（OR token，top 40）
 │    ├─ RRF 融合（BM25_WEIGHT）
 │    └─ 精确词加成（查询词原样出现 +EXACT_BONUS）
 │
 ├─ rerank：一次批量调用，让模型给片段打分并排序
 │
 ├─ 模式：
 │    ├─ fast：取 top 6 片段拼上下文 → 生成
 │    └─ precise：
 │         1) 用强查询词锁定包含该词的页
 │         2) 模型在候选页里挑最多 3 页
 │         3) 把整页原文交给模型精读
 │
 └─ generate：SSE 输出
      sources → token* → verify → done
```

**引用核对**（`retrieval/verify.py`）：把答案里带 `[n]` 的句子去掉标记后，与第 n 个来源做 token 重叠率计算，≥0.45 标「已核对」，否则标「未匹配」。这是保守的文本重合判断，不是语义蕴含。

**预览高亮**：

- 页面图：`/page/{page}/image?terms=...` 用 PyMuPDF `search_for` 找到矩形，加黄色 highlight 注释后渲染 PNG
- 文字预览：后端按版面重建 Markdown（标题/列表/公式/表格），前端渲染后用 TreeWalker 把命中词包成 `<mark>`

## 存储

| 路径 | 内容 |
| --- | --- |
| `backend/chroma_db/` | Chroma 持久化向量库 |
| `backend/data/` | 原始文档 + `documents.json` |
| `backend/data/fts.db` | SQLite FTS5 关键词索引 |
| `backend/data/vocab.db` | 结构化词条 |
| `backend/data/conversations.db` | 对话与消息 |
| `backend/data/usage.db` | 用量估算 |
| `backend/cache/` | QA 缓存（按知识库版本失效） |
| `backend/exports/` | 手动导出的 Markdown |

## 前端

- React 19 + TypeScript + Tailwind 4，Vite 构建
- `api.ts`：SSE 按 `\n\n` 分帧、JSON 解析；封装文档/搜索/对话/导出接口
- `DocumentViewer`：原文图（可带高亮）/ 文字预览（Markdown + DOM 高亮）/ 命中片段
- 对话历史、会话切换、引用核对徽标、对话管理都在 `ChatPanel` 内完成

## 设计取舍

- **本地向量模型**：不依赖云端 embedding，离线可用、零费用；代价是首次要下 100 MB 模型
- **FTS5 而不是 BM25 库**：少一个重依赖，SQLite 自带，支持中文二元组和 doc_id 过滤
- **精准模式按页读**：比“多召回几个片段”更贵，但查定义、单词、公式时页码和原文更可靠
- **无鉴权**：定位是单机个人工具，默认只监听 127.0.0.1；要开放必须先加访问控制
