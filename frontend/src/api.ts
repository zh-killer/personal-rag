export interface Source {
  index: number
  chunk_id: string
  doc_id: string
  source: string
  page: number | null
  snippet: string
  terms?: string[]
}

export interface DocInfo {
  doc_id: string
  source: string
  chunk_count: number
  page_count: number
  size_bytes: number
  indexed_at: number
  is_current: boolean
}

export interface Health {
  status: string
  version: string
  llm_configured: boolean
  model: string
  embedding: string
  documents: number
  chunks: number
  fts_ready: boolean
}

export interface Stats {
  document_count: number
  chunk_count: number
  fts_ready: boolean
  embedding: string
  model: string
  embedding_provider: string
  chunk_size: number
  top_k: number
  supported_formats: string[]
  max_upload_size_mb: number
}

export interface ChatTurn {
  question: string
  answer: string
}

export interface SearchHit {
  chunk_id: string
  doc_id: string
  source: string
  page: number | null
  snippet: string
  score: number
  exact_hits?: number
  matched_terms: string[]
  match?: 'exact' | 'semantic'
}

export interface SearchResult {
  query: string
  doc_id: string | null
  count: number
  hits: SearchHit[]
  related: SearchHit[]
  terms: string[]
}

export interface UsageBreakdown {
  mode: string
  questions: number
  estimated_tokens: number
  model_calls: number
}

export interface UsageSummary {
  days: number
  questions: number
  estimated_tokens: number
  model_calls: number
  avg_elapsed_ms: number
  by_mode: UsageBreakdown[]
  daily: { day: string; questions: number; estimated_tokens: number; model_calls: number }[]
}

export interface PageText {
  doc_id: string
  source: string
  page: number
  text: string
  markdown?: string
  kind?: string
}

export interface CitationCheck {
  index: number
  cited: boolean
  verified: boolean
  overlap: number
}

export type StreamEvent =
  | { type: 'sources'; sources: Source[]; confidence: number }
  | { type: 'token'; text: string }
  | { type: 'verify'; checks: CitationCheck[]; unverified: number[] }
  | { type: 'done'; confidence: number; elapsed_ms: number; cached: boolean }
  | { type: 'error'; message: string }

export interface UploadAccepted {
  task_id: string
  status: string
  filename: string
  size_bytes: number
}

export interface IngestTask {
  id: string
  filename: string
  status: 'queued' | 'running' | 'done' | 'error'
  stage: string
  progress: number
  message: string
  doc_id: string
  chunk_count: number
  error: string
}

export interface ConversationSummary {
  id: string
  title: string
  created_at: number
  updated_at: number
  message_count: number
}

export interface ConversationCreated {
  id: string
  title: string
  created_at: number
  updated_at: number
}

export interface StoredMessage {
  id?: number
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  confidence?: number | null
  checks?: CitationCheck[]
}

export interface ConversationDetail {
  id: string
  title: string
  created_at: number
  updated_at: number
  messages: StoredMessage[]
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const data = (await response.json()) as { detail?: unknown }
    if (typeof data.detail === 'string') return data.detail
    return JSON.stringify(data.detail ?? data)
  } catch {
    return `HTTP ${response.status}`
  }
}

export async function fetchHealth(): Promise<Health> {
  const response = await fetch('/api/health', { cache: 'no-store' })
  if (!response.ok) throw new Error(await errorMessage(response))
  return (await response.json()) as Health
}

export async function fetchDocuments(): Promise<DocInfo[]> {
  const response = await fetch('/api/documents', { cache: 'no-store' })
  if (!response.ok) throw new Error(await errorMessage(response))
  const data = (await response.json()) as { documents: DocInfo[] }
  return data.documents
}

export async function fetchStats(): Promise<Stats> {
  const response = await fetch('/api/stats', { cache: 'no-store' })
  if (!response.ok) throw new Error(await errorMessage(response))
  return (await response.json()) as Stats
}

export async function fetchUsage(days = 7): Promise<UsageSummary> {
  const response = await fetch(`/api/usage/summary?days=${days}`, { cache: 'no-store' })
  if (!response.ok) throw new Error(await errorMessage(response))
  return (await response.json()) as UsageSummary
}

export async function uploadFile(file: File): Promise<UploadAccepted> {
  const form = new FormData()
  form.append('file', file)
  const response = await fetch('/api/upload', { method: 'POST', body: form })
  if (!response.ok) throw new Error(await errorMessage(response))
  return (await response.json()) as UploadAccepted
}

export async function fetchTask(taskId: string): Promise<IngestTask> {
  const response = await fetch(`/api/tasks/${encodeURIComponent(taskId)}`, {
    cache: 'no-store',
  })
  if (!response.ok) throw new Error(await errorMessage(response))
  return (await response.json()) as IngestTask
}

/** 轮询入库任务，直到完成或失败（最长 15 分钟） */
export async function waitForTask(
  taskId: string,
  onProgress?: (task: IngestTask) => void,
  intervalMs = 700,
): Promise<IngestTask> {
  const deadline = Date.now() + 15 * 60 * 1000
  while (Date.now() < deadline) {
    const task = await fetchTask(taskId)
    onProgress?.(task)
    if (task.status === 'done' || task.status === 'error') return task
    await new Promise((resolve) => setTimeout(resolve, intervalMs))
  }
  throw new Error('入库超时（超过 15 分钟），请查看后端日志')
}

export async function deleteDocument(docId: string): Promise<void> {
  const response = await fetch(`/api/documents/${encodeURIComponent(docId)}`, {
    method: 'DELETE',
  })
  if (!response.ok) throw new Error(await errorMessage(response))
}

export async function fetchConversations(query = ''): Promise<ConversationSummary[]> {
  const suffix = query.trim() ? `?q=${encodeURIComponent(query.trim())}` : ''
  const response = await fetch(`/api/conversations${suffix}`, { cache: 'no-store' })
  if (!response.ok) throw new Error(await errorMessage(response))
  const data = (await response.json()) as { conversations: ConversationSummary[] }
  return data.conversations
}

export async function createConversation(title: string): Promise<ConversationCreated> {
  const response = await fetch('/api/conversations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  })
  if (!response.ok) throw new Error(await errorMessage(response))
  return (await response.json()) as ConversationCreated
}

export async function fetchConversation(id: string): Promise<ConversationDetail> {
  const response = await fetch(`/api/conversations/${encodeURIComponent(id)}`, {
    cache: 'no-store',
  })
  if (!response.ok) throw new Error(await errorMessage(response))
  return (await response.json()) as ConversationDetail
}

export async function appendMessage(
  conversationId: string,
  message: StoredMessage,
): Promise<number> {
  const response = await fetch(
    `/api/conversations/${encodeURIComponent(conversationId)}/messages`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(message),
    },
  )
  if (!response.ok) throw new Error(await errorMessage(response))
  const data = (await response.json()) as { id: number }
  return data.id
}

export async function deleteMessages(
  conversationId: string,
  ids: number[],
): Promise<number> {
  const response = await fetch(
    `/api/conversations/${encodeURIComponent(conversationId)}/messages/delete`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids }),
    },
  )
  if (!response.ok) throw new Error(await errorMessage(response))
  const data = (await response.json()) as { removed: number }
  return data.removed
}

export async function renameConversation(id: string, title: string): Promise<void> {
  const response = await fetch(`/api/conversations/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  })
  if (!response.ok) throw new Error(await errorMessage(response))
}

export function conversationExportUrl(id: string): string {
  return `/api/conversations/${encodeURIComponent(id)}/export`
}

export async function saveConversationExport(
  id: string,
): Promise<{ path: string; filename: string; bytes: number }> {
  const response = await fetch(
    `/api/conversations/${encodeURIComponent(id)}/export/save`,
    { method: 'POST' },
  )
  if (!response.ok) throw new Error(await errorMessage(response))
  return (await response.json()) as { path: string; filename: string; bytes: number }
}

export async function deleteConversation(id: string): Promise<void> {
  const response = await fetch(`/api/conversations/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  })
  if (!response.ok) throw new Error(await errorMessage(response))
}

/** 原文检索：不走大模型，直接返回命中的片段与页码 */
export async function searchChunks(
  query: string,
  docId?: string,
  topK = 10,
): Promise<SearchResult> {
  const response = await fetch('/api/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, doc_id: docId ?? null, top_k: topK }),
  })
  if (!response.ok) throw new Error(await errorMessage(response))
  return (await response.json()) as SearchResult
}

export async function fetchPageText(docId: string, page: number): Promise<PageText> {
  const response = await fetch(
    `/api/documents/${encodeURIComponent(docId)}/page/${page}`,
    { cache: 'no-store' },
  )
  if (!response.ok) throw new Error(await errorMessage(response))
  return (await response.json()) as PageText
}

export function documentFileUrl(docId: string, page?: number | null): string {
  const suffix = page ? `#page=${page}&zoom=page-width` : ''
  return `/api/documents/${encodeURIComponent(docId)}/file${suffix}`
}

export type AnswerMode = 'auto' | 'fast' | 'precise'

/**
 * 读取 SSE 流。每个事件是一行 `data: {json}`，以空行分隔。
 * 注意：必须按块缓冲，否则一个 JSON 被拆到两个网络包里就会解析失败。
 */
export async function* streamQuery(
  question: string,
  history: ChatTurn[],
  signal: AbortSignal,
  docId?: string,
  mode: AnswerMode = 'auto',
): AsyncGenerator<StreamEvent> {
  const response = await fetch('/api/query/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, history, doc_id: docId ?? null, mode }),
    signal,
  })
  if (!response.ok) throw new Error(await errorMessage(response))
  if (!response.body) throw new Error('浏览器不支持流式响应')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let boundary = buffer.indexOf('\n\n')
    while (boundary >= 0) {
      const frame = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      for (const line of frame.split('\n')) {
        if (!line.startsWith('data: ')) continue
        try {
          yield JSON.parse(line.slice(6)) as StreamEvent
        } catch {
          // 半帧/心跳：忽略，继续读下一个包
        }
      }
      boundary = buffer.indexOf('\n\n')
    }
  }
}
