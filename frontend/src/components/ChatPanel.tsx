import { useEffect, useRef, useState } from 'react'
import { FileText, History, Loader2, Pencil, RotateCcw, Search, Send, Sparkles, Square, Trash2, Download, X } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import {
  appendMessage,
  conversationExportUrl,
  createConversation,
  deleteConversation,
  deleteMessages,
  fetchConversation,
  fetchConversations,
  renameConversation,
  saveConversationExport,
  streamQuery,
} from '../api'
import type {
  ChatTurn,
  CitationCheck,
  ConversationSummary,
  DocInfo,
  Source,
  StoredMessage,
} from '../api'
import DocumentViewer from './DocumentViewer'

interface Message {
  id: string
  serverId?: number
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  checks?: CitationCheck[]
  confidence?: number
  elapsedMs?: number
  cached?: boolean
  stopped?: boolean
  error?: string
}

interface Props {
  llmConfigured: boolean
  documentCount: number
  documents: DocInfo[]
  onGoUpload: () => void
}

const SUGGESTIONS = [
  '这个知识库里有哪些文档？',
  '帮我总结知识库的核心内容',
  '文档里有哪些容易混淆的概念？',
]

const STORAGE_KEY = 'personalrag.chat.v1'
const MAX_STORED = 60
const HISTORY_TURNS = 5
const NO_TERMS: string[] = []

function newId(): string {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2)
}

/** 从 localStorage 恢复上次会话（刷新/切页不丢） */
function loadStoredMessages(): Message[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed: unknown = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed
      .filter((item): item is Message => {
        if (!item || typeof item !== 'object') return false
        const message = item as Partial<Message>
        return (
          typeof message.id === 'string' &&
          typeof message.content === 'string' &&
          (message.role === 'user' || message.role === 'assistant')
        )
      })
      .slice(-MAX_STORED)
  } catch {
    return []
  }
}

function buildHistory(messages: Message[]): ChatTurn[] {
  const turns: ChatTurn[] = []
  for (let i = 0; i < messages.length - 1; i += 1) {
    const current = messages[i]
    const next = messages[i + 1]
    if (current.role === 'user' && next.role === 'assistant' && !next.error) {
      turns.push({ question: current.content, answer: next.content })
    }
  }
  return turns.slice(-HISTORY_TURNS)
}

function toMessages(items: StoredMessage[]): Message[] {
  return items
    .filter((item) => item.role === 'user' || item.role === 'assistant')
    .map((item) => ({
      id: newId(),
      serverId: item.id,
      role: item.role,
      content: item.content,
      sources: item.sources,
      confidence: item.confidence ?? undefined,
    }))
}

export default function ChatPanel({
  llmConfigured,
  documentCount,
  documents,
  onGoUpload,
}: Props) {
  const [messages, setMessages] = useState<Message[]>(loadStoredMessages)
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [nearBottom, setNearBottom] = useState(true)
  const [docId, setDocId] = useState('')
  const [mode, setMode] = useState<'auto' | 'fast' | 'precise'>('auto')
  const [conversationId, setConversationId] = useState('')
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [historyOpen, setHistoryOpen] = useState(false)
  const [historyQuery, setHistoryQuery] = useState('')
  const [hydrated, setHydrated] = useState(false)
  const [viewer, setViewer] = useState<Source | null>(null)
  const [renamingId, setRenamingId] = useState('')
  const [renameValue, setRenameValue] = useState('')
  const [confirmId, setConfirmId] = useState('')
  const [confirmRoundId, setConfirmRoundId] = useState('')
  const [actionError, setActionError] = useState('')
  const [actionNotice, setActionNotice] = useState('')
  const [exportPreview, setExportPreview] = useState<{ id: string; content: string } | null>(null)
  const [savedPath, setSavedPath] = useState('')
  const exportTextRef = useRef<HTMLTextAreaElement>(null)

  const containerRef = useRef<HTMLDivElement>(null)
  const endRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  const patch = (id: string, changes: (message: Message) => Partial<Message>) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...changes(m) } : m)))
  }

  const refreshList = async (query = historyQuery) => {
    try {
      setConversations(await fetchConversations(query))
    } catch {
      // 忽略
    }
  }

  useEffect(() => {
    if (!historyOpen) return
    const timer = setTimeout(() => {
      void refreshList(historyQuery)
    }, 250)
    return () => clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [historyOpen, historyQuery])

  // 启动时恢复最近一次对话；服务端没有记录时，把浏览器里的旧会话迁移过去
  useEffect(() => {
    let cancelled = false

    const hydrate = async () => {
      try {
        const list = await fetchConversations()
        if (cancelled) return
        setConversations(list)

        if (list.length > 0) {
          const detail = await fetchConversation(list[0].id)
          if (cancelled) return
          setConversationId(detail.id)
          setMessages(toMessages(detail.messages))
          return
        }

        const stored = loadStoredMessages()
        if (stored.length === 0) return
        const title =
          stored.find((item) => item.role === 'user')?.content.slice(0, 30) || '历史对话'
        const conversation = await createConversation(title)
        for (const item of stored) {
          await appendMessage(conversation.id, {
            role: item.role,
            content: item.content,
            sources: item.sources,
            confidence: item.confidence,
          })
        }
        if (cancelled) return
        setConversationId(conversation.id)
        setMessages(stored)
        setConversations([{ ...conversation, message_count: stored.length }])
        window.localStorage.removeItem(STORAGE_KEY)
      } catch {
        // 服务端不可用：不影响聊天，只是暂时没有历史
      } finally {
        if (!cancelled) setHydrated(true)
      }
    }

    void hydrate()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (nearBottom) endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, nearBottom])

  const handleScroll = () => {
    const el = containerRef.current
    if (!el) return
    setNearBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 120)
  }

  const ensureConversation = async (title: string): Promise<string> => {
    if (conversationId) return conversationId
    const conversation = await createConversation(title.slice(0, 30))
    setConversationId(conversation.id)
    setConversations((prev) => [{ ...conversation, message_count: 0 }, ...prev])
    return conversation.id
  }

  const send = async (raw?: string, historyOverride?: ChatTurn[]) => {
    const question = (raw ?? input).trim()
    if (!question || busy) return

    const history = historyOverride ?? buildHistory(messages)
    const userMessage: Message = { id: newId(), role: 'user', content: question }
    const assistantId = newId()
    const assistantMessage: Message = { id: assistantId, role: 'assistant', content: '' }

    setMessages((prev) => [...prev, userMessage, assistantMessage])
    setInput('')
    setBusy(true)
    setNearBottom(true)

    let answerText = ''
    let answerSources: Source[] = []
    let answerConfidence: number | undefined
    let failure = ''
    let activeId = conversationId

    try {
      activeId = await ensureConversation(question)
      const userServerId = await appendMessage(activeId, {
        role: 'user',
        content: question,
      })
      if (userServerId) patch(userMessage.id, () => ({ serverId: userServerId }))
    } catch {
      // 历史保存失败不影响本轮问答
    }

    const controller = new AbortController()
    abortRef.current = controller

    try {
      for await (const event of streamQuery(
        question,
        history,
        controller.signal,
        docId || undefined,
        mode,
      )) {
        if (event.type === 'sources') {
          answerSources = event.sources
          answerConfidence = event.confidence
          patch(assistantId, () => ({
            sources: event.sources,
            confidence: event.confidence,
          }))
        } else if (event.type === 'token') {
          answerText += event.text
          patch(assistantId, () => ({ content: answerText }))
        } else if (event.type === 'verify') {
          patch(assistantId, () => ({ checks: event.checks }))
        } else if (event.type === 'done') {
          answerConfidence = event.confidence
          patch(assistantId, () => ({
            confidence: event.confidence,
            elapsedMs: event.elapsed_ms,
            cached: event.cached,
          }))
        } else if (event.type === 'error') {
          throw new Error(event.message)
        }
      }
    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        patch(assistantId, () => ({
          stopped: true,
          content: answerText || '（已停止）',
        }))
      } else {
        failure = (error as Error).message
        patch(assistantId, () => ({ error: failure }))
      }
    } finally {
      setBusy(false)
      abortRef.current = null
    }

    if (activeId) {
      const content = failure ? `请求失败：${failure}` : answerText || '（已停止）'
      try {
        const assistantServerId = await appendMessage(activeId, {
          role: 'assistant',
          content,
          sources: answerSources,
          confidence: answerConfidence,
        })
        if (assistantServerId) {
          patch(assistantId, () => ({ serverId: assistantServerId }))
        }
        await refreshList()
      } catch {
        // 忽略
      }
    }
  }

  const stop = () => abortRef.current?.abort()

  const reset = () => {
    if (busy) return
    setMessages([])
    setInput('')
    setConversationId('')
    setHistoryOpen(false)
    try {
      window.localStorage.removeItem(STORAGE_KEY)
    } catch {
      // 忽略
    }
  }

  const openConversation = async (id: string) => {
    if (busy) return
    try {
      const detail = await fetchConversation(id)
      setConversationId(detail.id)
      setMessages(toMessages(detail.messages))
      setHistoryOpen(false)
      setNearBottom(true)
    } catch {
      // 忽略
    }
  }

  const showError = (message: string) => {
    setActionError(message)
    window.setTimeout(() => setActionError(''), 5000)
  }

  const showNotice = (message: string) => {
    setActionNotice(message)
    window.setTimeout(() => setActionNotice(''), 4000)
  }

  const beginRename = (id: string, title: string) => {
    setConfirmId('')
    setRenamingId(id)
    setRenameValue(title)
  }

  const submitRename = async () => {
    if (!renamingId) return
    const next = renameValue.trim()
    if (!next) {
      setRenamingId('')
      return
    }
    try {
      await renameConversation(renamingId, next)
      setRenamingId('')
      await refreshList()
    } catch (error) {
      showError(`重命名失败：${(error as Error).message}`)
    }
  }

  const removeConversation = async (id: string) => {
    if (busy) return
    try {
      await deleteConversation(id)
      setConfirmId('')
      await refreshList()
      if (id === conversationId) {
        setConversationId('')
        setMessages([])
      }
    } catch (error) {
      showError(`删除失败：${(error as Error).message}`)
    }
  }

  const requestDelete = (id: string) => {
    setRenamingId('')
    setConfirmId(id)
    window.setTimeout(() => {
      setConfirmId((current) => (current === id ? '' : current))
    }, 4000)
  }

  const openExport = async (id: string) => {
    try {
      const response = await fetch(conversationExportUrl(id))
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      setSavedPath('')
      setExportPreview({ id, content: await response.text() })
    } catch (error) {
      showError(`导出失败：${(error as Error).message}`)
    }
  }

  const copyExport = async () => {
    if (!exportPreview) return
    try {
      await navigator.clipboard.writeText(exportPreview.content)
      showNotice('已复制到剪贴板')
    } catch {
      exportTextRef.current?.focus()
      exportTextRef.current?.select()
      showError('浏览器不允许自动复制，已帮你全选，按 Ctrl+C 即可')
    }
  }

  const saveToDisk = async () => {
    if (!exportPreview) return
    try {
      const result = await saveConversationExport(exportPreview.id)
      setSavedPath(result.path)
      showNotice('已保存到本机')
    } catch (error) {
      showError(`保存失败：${(error as Error).message}`)
    }
  }

  const downloadExport = () => {
    if (!exportPreview) return
    const url = URL.createObjectURL(
      new Blob([exportPreview.content], { type: 'text/markdown;charset=utf-8' }),
    )
    const link = document.createElement('a')
    link.href = url
    link.download = `personalrag-chat-${exportPreview.id}.md`
    document.body.appendChild(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(url)
    showNotice('已触发下载；若浏览器拦截，请直接复制预览内容')
  }

  const deleteRound = async (message: Message) => {
    if (busy) return
    const index = messages.findIndex((item) => item.id === message.id)
    if (index < 0) return

    const pair = [messages[index - 1], messages[index]].filter(Boolean)
    const ids = pair
      .map((item) => item.serverId)
      .filter((value): value is number => typeof value === 'number')

    if (!conversationId || ids.length === 0) {
      showError('这条消息还没有同步到服务端，暂时不能删除')
      return
    }

    try {
      await deleteMessages(conversationId, ids)
      setConfirmRoundId('')
      setMessages((prev) => prev.filter((item) => !pair.some((p) => p.id === item.id)))
      await refreshList()
    } catch (error) {
      showError(`删除失败：${(error as Error).message}`)
    }
  }

  const requestDeleteRound = (id: string) => {
    setConfirmRoundId(id)
    window.setTimeout(() => {
      setConfirmRoundId((current) => (current === id ? '' : current))
    }, 4000)
  }

  const regenerate = async (message: Message) => {
    if (busy) return
    const index = messages.findIndex((item) => item.id === message.id)
    if (index <= 0) return

    const questionMessage = messages[index - 1]
    if (questionMessage.role !== 'user') return

    const pair = [questionMessage, messages[index]]
    const ids = pair
      .map((item) => item.serverId)
      .filter((value): value is number => typeof value === 'number')

    if (conversationId && ids.length > 0) {
      try {
        await deleteMessages(conversationId, ids)
      } catch (error) {
        showError(`重新生成失败：${(error as Error).message}`)
        return
      }
    }

    setConfirmRoundId('')
    const remaining = messages.filter((item) => !pair.some((p) => p.id === item.id))
    setMessages(remaining)
    await refreshList()
    await send(questionMessage.content, buildHistory(remaining))
  }

  useEffect(() => {
    if (exportPreview && exportTextRef.current) {
      exportTextRef.current.focus()
      exportTextRef.current.select()
    }
  }, [exportPreview])

  const disabled = !llmConfigured || busy
  const contextTurns = buildHistory(messages).length

  return (
    <div className="flex h-full flex-col">
      <div className="mb-3 flex items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2 text-sm text-slate-500">
          {documentCount > 0 ? (
            <span>
              已接入 <strong className="text-slate-700">{documentCount}</strong> 个文档
            </span>
          ) : (
            <button onClick={onGoUpload} className="text-purple-600 underline underline-offset-2">
              知识库为空，先上传文档
            </button>
          )}
          {contextTurns > 0 && (
            <span
              className="rounded-full bg-purple-50 px-2 py-0.5 text-[11px] text-purple-600"
              title="最近几轮问答会一起发给模型，刷新后也会保留"
            >
              记忆 {contextTurns} 轮
            </span>
          )}
          {documents.length > 0 && (
            <label className="flex items-center gap-1 text-[11px] text-slate-400">
              <span className="hidden sm:inline">范围</span>
              <select
                value={docId}
                onChange={(event) => setDocId(event.target.value)}
                className="h-6 max-w-[10rem] rounded-md border border-slate-200 bg-white px-1 text-[11px] text-slate-600 outline-none focus:border-purple-300"
                title="限定后只在这篇文档里检索"
              >
                <option value="">全部文档</option>
                {documents.map((doc) => (
                  <option key={doc.doc_id} value={doc.doc_id}>
                    {doc.source}
                  </option>
                ))}
              </select>
            </label>
          )}
          <div className="flex items-center rounded-lg border border-slate-200 bg-white p-0.5 text-[11px]">
            {(
              [
                { key: 'auto', label: '自动', title: '按问题自动选择：查细节走精准，总结对比走快速' },
                { key: 'fast', label: '快速', title: '向量 + 关键词召回片段后回答，快' },
                { key: 'precise', label: '精准', title: '大模型先挑页，再通读整页原文回答，更准' },
              ] as const
            ).map((item) => (
              <button
                key={item.key}
                onClick={() => setMode(item.key)}
                className={`rounded px-1.5 py-0.5 transition ${
                  mode === item.key
                    ? 'bg-purple-600 text-white'
                    : 'text-slate-500 hover:text-slate-800'
                }`}
                title={item.title}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
        <div className="relative flex shrink-0 items-center gap-3">
          {conversationId && (
            <button
              type="button"
              onClick={() => void openExport(conversationId)}
              className="flex items-center gap-1 text-xs text-slate-500 hover:text-slate-800"
              title="导出当前对话"
            >
              <Download size={13} /> 导出
            </button>
          )}
          <button
            onClick={() => setHistoryOpen((open) => !open)}
            className="flex items-center gap-1 text-xs text-slate-500 hover:text-slate-800"
            title="查看历史对话"
          >
            <History size={13} /> 历史
          </button>
          {messages.length > 0 && (
            <button
              onClick={reset}
              disabled={busy}
              className="flex items-center gap-1 text-xs text-slate-500 hover:text-slate-800 disabled:opacity-40"
            >
              <RotateCcw size={13} /> 新对话
            </button>
          )}

          {historyOpen && (
            <div className="absolute right-0 top-7 z-30 w-72 overflow-hidden rounded-xl border border-slate-200 bg-white shadow-xl">
              <div className="border-b border-slate-100 px-3 py-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-slate-600">历史对话</span>
                  <button
                    onClick={() => {
                      reset()
                      setHistoryOpen(false)
                    }}
                    className="text-[11px] text-purple-600 hover:underline"
                  >
                    新对话
                  </button>
                </div>
                <div className="mt-2 flex items-center gap-1.5 rounded-lg border border-slate-200 px-2">
                  <Search size={12} className="text-slate-400" />
                  <input
                    value={historyQuery}
                    onChange={(event) => setHistoryQuery(event.target.value)}
                    placeholder="搜索标题或对话内容"
                    className="h-7 flex-1 bg-transparent text-xs text-slate-700 outline-none placeholder:text-slate-400"
                  />
                </div>
              </div>
              <div className="max-h-80 overflow-y-auto p-1.5">
                {conversations.length === 0 && (
                  <p className="px-2 py-6 text-center text-xs text-slate-400">还没有历史对话</p>
                )}
                {conversations.map((item) => (
                  <div
                    key={item.id}
                    className={`flex items-start gap-2 rounded-lg px-2 py-1.5 ${
                      item.id === conversationId ? 'bg-purple-50' : 'hover:bg-slate-50'
                    }`}
                  >
                    {renamingId === item.id ? (
                      <div className="flex min-w-0 flex-1 items-center gap-1">
                        <input
                          autoFocus
                          value={renameValue}
                          onChange={(event) => setRenameValue(event.target.value)}
                          onKeyDown={(event) => {
                            if (event.key === 'Enter') void submitRename()
                            if (event.key === 'Escape') setRenamingId('')
                          }}
                          className="h-6 min-w-0 flex-1 rounded border border-purple-300 px-1 text-xs text-slate-700 outline-none"
                        />
                        <button
                          type="button"
                          onClick={() => void submitRename()}
                          className="shrink-0 text-[11px] text-purple-600 hover:underline"
                        >
                          保存
                        </button>
                      </div>
                    ) : (
                      <button
                        type="button"
                        onClick={() => void openConversation(item.id)}
                        className="min-w-0 flex-1 text-left"
                      >
                        <p className="truncate text-xs text-slate-700">
                          {item.title || '未命名对话'}
                        </p>
                        <p className="text-[10px] text-slate-400">
                          {new Date(item.updated_at * 1000).toLocaleString('zh-CN', {
                            month: '2-digit',
                            day: '2-digit',
                            hour: '2-digit',
                            minute: '2-digit',
                          })}
                          {' · '}
                          {item.message_count} 条消息
                        </p>
                      </button>
                    )}

                    <div className="flex shrink-0 items-center gap-0.5">
                      <button
                        type="button"
                        onClick={() => beginRename(item.id, item.title)}
                        className="rounded p-1 text-slate-300 hover:text-slate-700"
                        aria-label="重命名"
                        title="重命名"
                      >
                        <Pencil size={12} />
                      </button>
                      <button
                        type="button"
                        onClick={() => void openExport(item.id)}
                        className="rounded p-1 text-slate-300 hover:text-slate-700"
                        aria-label="导出"
                        title="导出 Markdown"
                      >
                        <Download size={12} />
                      </button>
                      {confirmId === item.id ? (
                        <button
                          type="button"
                          onClick={() => void removeConversation(item.id)}
                          className="rounded bg-red-50 px-1.5 py-0.5 text-[10px] text-red-600"
                        >
                          确认删除
                        </button>
                      ) : (
                        <button
                          type="button"
                          onClick={() => requestDelete(item.id)}
                          className="rounded p-1 text-slate-300 hover:text-red-500"
                          aria-label="删除对话"
                          title="删除对话"
                        >
                          <Trash2 size={12} />
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {actionError && (
        <p className="mb-2 rounded-lg bg-red-50 px-3 py-1.5 text-[11px] text-red-600">
          {actionError}
        </p>
      )}

      {actionNotice && (
        <p className="mb-2 rounded-lg bg-emerald-50 px-3 py-1.5 text-[11px] text-emerald-700">
          {actionNotice}
        </p>
      )}

      {!llmConfigured && (
        <div className="mb-3 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
          后端没有检测到 LLM API Key。请编辑 <code>backend/.env</code> 设置{' '}
          <code>OPENAI_API_KEY</code>，然后重启 <code>python run.py</code>。
        </div>
      )}

      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="flex-1 space-y-4 overflow-y-auto rounded-2xl bg-slate-50/60 p-3 sm:p-4"
      >
        {!hydrated && messages.length === 0 && (
          <div className="flex h-full items-center justify-center py-10 text-slate-400">
            <Loader2 size={20} className="animate-spin" />
          </div>
        )}

        {messages.length === 0 && hydrated && (
          <div className="flex h-full flex-col items-center justify-center gap-4 py-10 text-center">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-purple-100 text-purple-600">
              <Sparkles size={26} />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-slate-800">向你的知识库提问</h2>
              <p className="mt-1 text-sm text-slate-500">
                回答只基于已索引的文档，并标注来源与页码
              </p>
            </div>
            <div className="flex max-w-lg flex-wrap justify-center gap-2">
              {SUGGESTIONS.map((item) => (
                <button
                  key={item}
                  onClick={() => void send(item)}
                  disabled={disabled}
                  className="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs text-slate-600 transition hover:border-purple-300 hover:text-purple-700 disabled:opacity-50"
                >
                  {item}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          <div
            key={message.id}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[88%] rounded-2xl px-4 py-3 shadow-sm sm:max-w-[80%] ${
                message.role === 'user'
                  ? 'bg-purple-600 text-white'
                  : 'border border-slate-200 bg-white text-slate-800'
              }`}
            >
              {message.role === 'user' ? (
                <p className="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</p>
              ) : message.error ? (
                <p className="text-sm text-red-600">请求失败：{message.error}</p>
              ) : (
                <>
                  {message.content ? (
                    <div className="md-body">
                      <ReactMarkdown>{message.content}</ReactMarkdown>
                    </div>
                  ) : (
                    <span className="flex items-center gap-2 text-sm text-slate-400">
                      <Loader2 size={14} className="animate-spin" />
                      正在检索与生成…
                    </span>
                  )}

                  {message.sources && message.sources.length > 0 && (
                    <div className="mt-3 space-y-1.5 border-t border-slate-100 pt-2.5">
                      <p className="flex items-center gap-1 text-[11px] font-medium uppercase tracking-wide text-slate-400">
                        <FileText size={12} /> 参考来源
                      </p>
                      {message.sources.map((source) => {
                        const check = message.checks?.find(
                          (item) => item.index === source.index,
                        )
                        return (
                          <div
                            key={`${message.id}-${source.index}`}
                            role="button"
                            tabIndex={0}
                            onClick={() => setViewer(source)}
                            onKeyDown={(event) => {
                              if (event.key === 'Enter' || event.key === ' ') setViewer(source)
                            }}
                            className="cursor-pointer rounded-lg bg-slate-50 px-2.5 py-1.5 text-xs text-slate-600 transition hover:bg-purple-50"
                            title="点击查看原文这一页"
                          >
                            <div className="flex items-center justify-between gap-2">
                              <span className="font-medium text-slate-700">
                                [{source.index}] {source.source}
                                {source.page ? (
                                  <span className="ml-1 font-normal text-slate-400">
                                    第 {source.page} 页
                                  </span>
                                ) : null}
                              </span>
                              {check?.cited ? (
                                check.verified ? (
                                  <span className="shrink-0 text-[10px] text-emerald-600">已核对</span>
                                ) : (
                                  <span className="shrink-0 text-[10px] text-amber-600">未匹配</span>
                                )
                              ) : null}
                            </div>
                            <p className="mt-0.5 line-clamp-2 text-[11px] leading-relaxed text-slate-500">
                              {source.snippet}
                            </p>
                          </div>
                        )
                      })}
                    </div>
                  )}

                  {(message.confidence !== undefined || message.stopped || message.cached) && (
                    <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-slate-400">
                      {message.confidence !== undefined && (
                        <span>置信度 {Math.round(message.confidence * 100)}%</span>
                      )}
                      {message.confidence !== undefined && message.confidence < 0.45 && (
                        <span className="text-amber-600">把握不大，可到「搜索」页直接搜原文</span>
                      )}
                      {message.elapsedMs !== undefined && <span>· {(message.elapsedMs / 1000).toFixed(1)}s</span>}
                      {message.cached && (
                        <span className="rounded bg-emerald-50 px-1.5 py-0.5 text-emerald-600">缓存命中</span>
                      )}
                      {message.stopped && (
                        <span className="rounded bg-amber-50 px-1.5 py-0.5 text-amber-600">已停止</span>
                      )}
                    </div>
                  )}

                  {message.checks?.some((check) => check.cited && !check.verified) && (
                    <p className="mt-1 text-[11px] text-amber-600">
                      部分引用没能在原文中匹配，建议点开来源核对。
                    </p>
                  )}

                  {message.role === 'assistant' && !busy && (
                    <div className="mt-1.5 flex items-center gap-3 text-[11px] text-slate-400">
                      {index === messages.length - 1 && (
                        <button
                          type="button"
                          onClick={() => void regenerate(message)}
                          className="hover:text-purple-600"
                        >
                          重新生成
                        </button>
                      )}
                      {confirmRoundId === message.id ? (
                        <button
                          type="button"
                          onClick={() => void deleteRound(message)}
                          className="text-red-600"
                        >
                          确认删除这一轮
                        </button>
                      ) : (
                        <button
                          type="button"
                          onClick={() => requestDeleteRound(message.id)}
                          className="hover:text-red-500"
                        >
                          删除这一轮
                        </button>
                      )}
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <div className="mt-3 rounded-2xl border border-slate-200 bg-white p-2">
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                void send()
              }
            }}
            rows={1}
            placeholder={
              llmConfigured ? '输入问题，Enter 发送，Shift+Enter 换行' : '未配置 API Key，暂不可提问'
            }
            disabled={!llmConfigured}
            className="max-h-32 min-h-10 flex-1 resize-none bg-transparent px-2 py-2 text-sm text-slate-800 outline-none placeholder:text-slate-400 disabled:opacity-60"
          />
          {busy ? (
            <button
              onClick={stop}
              className="flex h-10 items-center gap-1.5 rounded-xl bg-slate-800 px-3 text-sm text-white transition hover:bg-slate-700"
            >
              <Square size={14} /> 停止
            </button>
          ) : (
            <button
              onClick={() => void send()}
              disabled={!input.trim() || !llmConfigured}
              className="flex h-10 w-10 items-center justify-center rounded-xl bg-purple-600 text-white transition hover:bg-purple-700 disabled:cursor-not-allowed disabled:opacity-40"
              aria-label="发送"
            >
              <Send size={16} />
            </button>
          )}
        </div>
        <p className="px-2 pb-1 pt-0.5 text-[11px] text-slate-400">
          回答基于已索引文档，引用格式为 [序号]；点来源可查看原文
        </p>
      </div>

      {viewer && (
        <DocumentViewer
          docId={viewer.doc_id}
          source={viewer.source}
          page={viewer.page}
          terms={viewer.terms && viewer.terms.length > 0 ? viewer.terms : NO_TERMS}
          snippet={viewer.snippet}
          onClose={() => setViewer(null)}
        />
      )}

      {exportPreview && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 p-4"
          onClick={() => setExportPreview(null)}
        >
          <div
            className="flex h-[80vh] w-full max-w-2xl flex-col overflow-hidden rounded-2xl bg-white shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-slate-200 px-4 py-2.5">
              <span className="text-sm font-medium text-slate-800">导出对话（Markdown）</span>
              <button
                type="button"
                onClick={() => setExportPreview(null)}
                className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
                aria-label="关闭"
              >
                <X size={18} />
              </button>
            </div>
            <textarea
              ref={exportTextRef}
              readOnly
              value={exportPreview.content}
              className="flex-1 resize-none bg-white p-4 font-mono text-[12px] leading-relaxed text-slate-700 outline-none"
            />

            {savedPath && (
              <div className="border-t border-emerald-200 bg-emerald-50 px-4 py-2 text-[11px] text-emerald-800">
                已保存到：<span className="select-all break-all font-mono">{savedPath}</span>
              </div>
            )}

            <div className="flex flex-wrap items-center justify-between gap-2 border-t border-slate-200 px-4 py-2">
              <span className="text-[11px] text-slate-400">保存位置：项目 backend/exports/</span>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => exportTextRef.current?.select()}
                  className="rounded-lg border border-slate-200 px-3 py-1.5 text-xs text-slate-600 hover:border-purple-300 hover:text-purple-700"
                >
                  全选
                </button>
                <button
                  type="button"
                  onClick={() => void copyExport()}
                  className="rounded-lg border border-slate-200 px-3 py-1.5 text-xs text-slate-600 hover:border-purple-300 hover:text-purple-700"
                >
                  复制
                </button>
                <button
                  type="button"
                  onClick={downloadExport}
                  className="rounded-lg border border-slate-200 px-3 py-1.5 text-xs text-slate-600 hover:border-purple-300 hover:text-purple-700"
                >
                  下载 .md
                </button>
                <button
                  type="button"
                  onClick={() => void saveToDisk()}
                  className="rounded-lg bg-purple-600 px-3 py-1.5 text-xs text-white hover:bg-purple-700"
                >
                  保存到本机
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
