import { useCallback, useEffect, useState } from 'react'
import {
  AlertCircle,
  Database,
  Download,
  FileText,
  Loader2,
  RefreshCw,
  Trash2,
  UploadCloud,
} from 'lucide-react'
import { deleteDocument, fetchDocuments, fetchStats, fetchUsage } from '../api'
import type { DocInfo, Stats, UsageSummary } from '../api'

interface Props {
  refreshTrigger: number
  onGoUpload: () => void
}

function formatSize(bytes: number): string {
  if (!bytes) return '-'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function formatTime(seconds: number): string {
  if (!seconds) return '-'
  return new Date(seconds * 1000).toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export default function DocListPanel({ refreshTrigger, onGoUpload }: Props) {
  const [docs, setDocs] = useState<DocInfo[]>([])
  const [stats, setStats] = useState<Stats | null>(null)
  const [usage, setUsage] = useState<UsageSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [deleting, setDeleting] = useState('')

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const [documents, statistics, usageSummary] = await Promise.all([
        fetchDocuments(),
        fetchStats(),
        fetchUsage(),
      ])
      setDocs(documents)
      setStats(statistics)
      setUsage(usageSummary)
      setError('')
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load, refreshTrigger])

  const remove = async (doc: DocInfo) => {
    if (!window.confirm(`确定删除「${doc.source}」及其 ${doc.chunk_count} 个索引片段吗？`)) return
    setDeleting(doc.doc_id)
    try {
      await deleteDocument(doc.doc_id)
      await load()
    } catch (err) {
      window.alert(`删除失败：${(err as Error).message}`)
    } finally {
      setDeleting('')
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24 text-slate-400">
        <Loader2 size={22} className="animate-spin" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="rounded-2xl border border-red-200 bg-red-50 px-5 py-6 text-sm text-red-700">
        <p className="flex items-center gap-2 font-medium">
          <AlertCircle size={16} /> 加载失败：{error}
        </p>
        <p className="mt-2 text-xs text-red-600">
          确认后端已启动（python run.py），并且浏览器访问的是同一地址。
        </p>
        <button onClick={() => void load()} className="mt-3 text-xs text-red-700 underline">
          重试
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-slate-800">已索引文档</h2>
          <p className="mt-0.5 text-xs text-slate-500">
            删除会同时移除源文件与向量，操作不可撤销
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => void load()}
            className="flex items-center gap-1 text-xs text-slate-500 hover:text-purple-600"
          >
            <RefreshCw size={13} /> 刷新
          </button>
          <button
            onClick={onGoUpload}
            className="flex items-center gap-1 rounded-lg bg-purple-600 px-3 py-1.5 text-xs text-white hover:bg-purple-700"
          >
            <UploadCloud size={13} /> 上传
          </button>
        </div>
      </div>

      {docs.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-slate-300 bg-white px-6 py-16 text-center">
          <Database size={40} className="mx-auto mb-3 text-slate-300" />
          <p className="text-sm text-slate-600">知识库还是空的</p>
          <button onClick={onGoUpload} className="mt-3 text-sm text-purple-600 underline">
            去上传第一份文档
          </button>
        </div>
      ) : (
        <div className="space-y-2">
          {docs.map((doc) => (
            <div
              key={doc.doc_id}
              className="flex items-center justify-between gap-3 rounded-xl border border-slate-200 bg-white px-4 py-3"
            >
              <div className="flex min-w-0 items-center gap-3">
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-purple-50 text-purple-600">
                  <FileText size={18} />
                </div>
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-slate-800">{doc.source}</p>
                  <p className="mt-0.5 text-xs text-slate-400">
                    {doc.chunk_count} 片段
                    {doc.page_count ? ` · ${doc.page_count} 页` : ''} · {formatSize(doc.size_bytes)} ·{' '}
                    {formatTime(doc.indexed_at)}
                  </p>
                </div>
              </div>

              <button
                onClick={() => void remove(doc)}
                disabled={deleting === doc.doc_id}
                className="shrink-0 rounded-lg p-2 text-slate-400 transition hover:bg-red-50 hover:text-red-500 disabled:opacity-40"
                aria-label={`删除 ${doc.source}`}
              >
                {deleting === doc.doc_id ? (
                  <Loader2 size={16} className="animate-spin" />
                ) : (
                  <Trash2 size={16} />
                )}
              </button>
            </div>
          ))}
        </div>
      )}

      {stats && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {[
            { label: '文档', value: stats.document_count },
            { label: '片段', value: stats.chunk_count },
            { label: '切块', value: `${stats.chunk_size} 字` },
            { label: '召回', value: `Top ${stats.top_k}` },
          ].map((card) => (
            <div key={card.label} className="rounded-xl border border-slate-200 bg-white px-4 py-3">
              <p className="text-lg font-semibold text-slate-800">{card.value}</p>
              <p className="text-xs text-slate-400">{card.label}</p>
            </div>
          ))}
        </div>
      )}

      {stats && (
        <p className="text-[11px] leading-relaxed text-slate-400">
          向量模型：{stats.embedding} · 生成模型：{stats.model} · 支持格式：
          {stats.supported_formats.join(' / ')} · 关键索引：{stats.fts_ready ? '已就绪' : '未就绪'}
        </p>
      )}

      <div className="rounded-xl border border-slate-200 bg-white p-4">
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-sm font-medium text-slate-700">数据与用量</h3>
          <span className="text-[11px] text-slate-400">
            近 {usage?.days ?? 7} 天（估算）
          </span>
        </div>

        {usage && usage.questions > 0 ? (
          <p className="mt-1 text-xs text-slate-500">
            {usage.questions} 次模型问答 · 约 {usage.estimated_tokens.toLocaleString()} tokens ·{' '}
            {usage.model_calls} 次模型调用 · 平均{' '}
            {(usage.avg_elapsed_ms / 1000).toFixed(1)}s
          </p>
        ) : (
          <p className="mt-1 text-xs text-slate-400">近 7 天还没有模型问答记录</p>
        )}

        <div className="mt-3 flex flex-wrap gap-2">
          <a
            href="/api/export/backup"
            className="flex items-center gap-1.5 rounded-lg bg-slate-800 px-3 py-1.5 text-xs text-white transition hover:bg-slate-700"
          >
            <Download size={13} /> 下载备份（zip）
          </a>
          <a
            href="/api/export/conversations"
            className="flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 text-xs text-slate-600 transition hover:border-purple-300 hover:text-purple-700"
          >
            <FileText size={13} /> 导出对话（Markdown）
          </a>
        </div>

        <p className="mt-2 text-[11px] leading-relaxed text-slate-400">
          备份包含原始文档与对话记录；恢复后会自动重建向量 / 关键词 / 词条索引。
        </p>
      </div>
    </div>
  )
}
