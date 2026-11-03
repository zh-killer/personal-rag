# PersonalRAG

[![tests](https://github.com/zh-killer/personal-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/zh-killer/personal-rag/actions/workflows/tests.yml)

本地优先的个人知识库问答系统：把课程笔记、单词表、扫描件丢进去，用自然语言提问，答案带**页码级来源**；也可以直接搜关键词，点结果跳到原文那一页。

索引、对话、词条全部存在本机；向量模型离线运行，只有生成答案时调用大模型 API。

## 亮点

- **混合检索**：Chroma 向量 + SQLite **FTS5** 关键词索引（中文按二元组切分），RRF 融合 + 精确词加成 + 大模型批量重排
- **三种回答模式**：自动 / 快速 / 精准。精准模式先让模型选页，再把整页原文交给模型精读，适合查定义、单词、公式
- **页码级溯源**：答案用 `[n]` 引用来源；点来源直接打开原文页面，**命中词自动标黄**（页面图按 PDF 坐标标，文字预览按 DOM 标）
- **引用核对**：把答案里的每个引用和原文片段做文本比对，标「已核对 / 未匹配」，减少“看起来有出处”的幻觉
- **扫描件 OCR**：离线 RapidOCR，扫描版 PDF 与图片自动识别；有文字层的页面不走 OCR
- **结构化词条**：词表类文档入库时自动抽成「单词 / 音标 / 词性 / 释义 / 页码」，查词约 0.01 秒，不消耗模型调用
- **排版还原**：文字预览按 PDF 版面重建 Markdown（标题、段落、列表、公式、表格），不是把切块拼成一堵墙
- **对话管理**：历史存在服务端，支持重命名、按内容搜索、删除单轮、重新生成、单条导出
- **备份与用量**：知识库打包 zip、对话导出 Markdown、近 7 天模型问答与 token 估算

## 快速开始

### 环境

- Python 3.10+（3.11 已验证）
- 一个 OpenAI 兼容的模型 API Key（默认 DeepSeek）
- **不需要 Node**：`frontend/dist` 已构建，后端直接托管

### Windows

```bat
cd backend
pip install -r requirements.txt
copy .env.example .env
:: 编辑 backend\.env，填入 OPENAI_API_KEY
```

回到项目根目录，双击 `start.bat`（或 `python backend/run.py --open`）。
浏览器会自动打开 <http://127.0.0.1:8000>。

### macOS / Linux

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# 编辑 backend/.env，填入 OPENAI_API_KEY
cd ..
python backend/run.py --open
```

首次启动会下载本地向量模型 `BAAI/bge-small-zh-v1.5`（约 100 MB），之后可离线运行。

### 只想改前端

```bash
cd frontend
npm install
npm run dev        # 开发模式，代理到 8000
npm run build      # 构建到 dist，后端直接托管
```

## 使用

1. **上传**：拖入 PDF / Markdown / TXT / 图片；上传立即返回，解析、OCR、向量化在后台进行，页面显示进度
2. **搜索**：输入关键词，只返回**真正包含该词**的片段；没有精确命中时会单独列出语义相关结果，并提示“知识库中没有出现该词”
3. **问答**：可以限定到某一篇文档；回答流式输出，可随时停止；多轮对话自动带最近 5 轮上下文
4. **看原文**：点来源或搜索结果 →「原文」显示整页渲染图（命中词标黄），「文字预览」显示排好版的 Markdown
5. **历史**：右上角「历史」可回看、搜索、重命名、删除、导出；回答下方可「删除这一轮 / 重新生成」

## 配置

`backend/.env`，除 API Key 外都有默认值。

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 空 | LLM Key（DeepSeek / OpenAI / 硅基流动等兼容接口），**必填** |
| `OPENAI_API_BASE` | `https://api.deepseek.com/v1` | 接口地址 |
| `LLM_MODEL` | `deepseek-chat` | 模型名 |
| `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` | 0.1 / 2048 | 生成参数 |
| `EMBEDDING_PROVIDER` | `local` | `local` 离线 / `api` / `auto` |
| `EMBEDDING_LOCAL_MODEL` | `BAAI/bge-small-zh-v1.5` | 本地向量模型（512 维） |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 500 / 60 | 切块大小与重叠 |
| `TOP_K` / `RERANK_TOP_K` | 10 / 6 | 召回数 / 重排后保留数 |
| `BM25_WEIGHT` | 0.45 | RRF 中关键词通道的权重 |
| `EXACT_BONUS` | 0.03 | 查询词原样出现时的加分 |
| `HOST` / `PORT` | 127.0.0.1 / 8000 | 监听地址与端口 |
| `MAX_UPLOAD_MB` | 50 | 单文件大小上限 |
| `OCR_ENABLED` | 1 | 扫描件 / 图片是否自动 OCR |
| `HISTORY_TURNS` | 5 | 对话记忆轮数 |
| `ENABLE_QA_CACHE` | 1 | 问答缓存（多轮请求不参与） |
| `DATA_DIR` / `CHROMA_DIR` / `CACHE_DIR` | `backend/` 下 | 数据、向量库、缓存目录 |

完整变量见 `backend/.env.example`。

## 项目结构

```
personal-rag/
├── .github/workflows/tests.yml  # CI：装最小测试依赖跑 pytest
├── start.bat                 # 一键启动
├── backend/
│   ├── run.py                # 启动器：依赖自检 / 端口探测 / 自动开浏览器
│   ├── requirements.txt      # 运行依赖
│   ├── requirements-test.txt # 单测依赖（不含 torch / 向量模型 / OCR）
│   ├── requirements-dev.txt  # 开发依赖（pytest / ruff）
│   ├── pytest.ini
│   ├── app/
│   │   ├── main.py           # 应用装配：lifespan / 中间件 / include_router / 静态托管
│   │   ├── deps.py           # 进程内单例（索引、检索组件、会话、用量）
│   │   ├── schemas.py        # 请求 / 响应模型
│   │   ├── api/              # 路由层
│   │   │   ├── documents.py  # 上传 / 入库任务 / 文档台账 / 原文与页面渲染
│   │   │   ├── query.py      # 关键词检索 / 问答 / SSE 流式
│   │   │   ├── conversations.py # 对话增删改查与导出
│   │   │   ├── system.py     # 健康检查 / 统计 / 缓存 / 用量 / 备份
│   │   │   └── common.py     # SSE 帧、上下文裁剪、片段截取等共用函数
│   │   ├── core/             # 配置、LLM 客户端
│   │   ├── indexing/         # 解析 / OCR / 切块 / 向量化 / FTS / 词条 / 版面还原
│   │   ├── retrieval/        # 路由 / 改写 / 召回 / 重排 / 生成 / 精准阅读 / 引用核对
│   │   ├── services/         # 后台任务 / 对话 / 用量 / 备份
│   │   └── cache/            # QA 缓存
│   ├── tests/                # 单元测试（分词 / 切块 / 引用核对 / SSE 帧）
│   ├── scripts/              # selftest.py / eval.py
│   ├── eval/                 # cases.json / report.md
│   └── data/                 # 上传的原始文档（git 忽略）
└── frontend/
    ├── dist/                 # 构建产物，后端直接托管
    └── src/                  # React + TypeScript + Tailwind
```

## 主要接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查、模型与索引状态 |
| GET / POST / DELETE | `/api/documents` | 文档列表 / 上传（202 + 任务 id）/ 删除 |
| GET | `/api/tasks/{task_id}` | 入库任务进度 |
| GET | `/api/search` | 关键词精确搜索 + 语义相关 |
| POST | `/api/query` | 非流式问答 |
| POST | `/api/query/stream` | SSE 流式问答（`sources → token* → verify → done`） |
| GET | `/api/documents/{id}/page/{page}` | 该页文字 + 版面重建 Markdown |
| GET | `/api/documents/{id}/page/{page}/image` | 该页渲染图，`terms` 参数用于高亮 |
| GET / POST / PATCH / DELETE | `/api/conversations...` | 对话列表 / 新建 / 重命名 / 删除 / 消息 / 导出 |
| GET | `/api/usage/summary` | 近 N 天用量估算 |
| GET | `/api/export/backup` | 知识库 zip（文档 + 对话） |
| GET | `/api/export/conversations` | 全部对话 Markdown |

## 评测

`backend/eval/cases.json` 维护了 18 条固定用例（12 条词条、5 条知识、1 条知识库元问题）。改动检索或生成后跑一遍，能看出是变好还是变差：

```bash
python backend/scripts/eval.py
python backend/scripts/eval.py --category vocab
python backend/scripts/eval.py --limit 5
```

当前基线：**18/18 通过**（vocab 12/12、knowledge 5/5、meta 1/1）；词条题走结构化直查，约 0.01 秒；知识题真实调用模型，约 1–3 秒。详见 `docs/EVALUATION.md`。

## 单元测试

评测要连真实模型和知识库，跑一次几十秒；单元测试只覆盖纯逻辑，一秒内跑完：

```bash
cd backend
pip install -r requirements-test.txt   # 只装单测需要的包，不拉向量模型和 OCR
python -m pytest                       # 21 个用例
```

覆盖的是最容易悄悄坏掉、但端到端评测又看不出来的地方：分词与疑问句关键词抽取、切块是否保留页码和 `doc_id`、引用核对的三种状态（已核对 / 未匹配 / 未引用）、SSE 帧遇到换行是否还能只占一行 `data:`。

CI（GitHub Actions）装的是同一份 `requirements-test.txt`，不下载 torch 和向量模型，整套约 20 秒；想更快可以本地跑 `pytest -q`。

## 已知限制

- **单用户、无鉴权**：默认只监听 `127.0.0.1`。如需局域网访问，必须自行加反向代理和访问控制
- **OCR 不是万能的**：清晰扫描件效果好；手写、低分辨率、复杂表格会掉字，重要文档建议人工校对
- **OCR 有速度成本**：每页约 1–3 秒，几十页的扫描件要走后台任务
- **公式上下标**可能丢失（PDF 文本层限制），文字预览里以代码块展示
- **扫描件页面图没有坐标高亮**：没有文字层就无法定位，只能看 OCR 文本
- 未实现图片理解（VLM）、Agent 多步检索、跨文档对比

## License

MIT，见 [LICENSE](LICENSE)。
