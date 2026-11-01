import { useState } from 'react'
import { FileText, Loader2, Search, Sparkles } from 'lucide-react'
import { searchChunks } from '../api'
import type { DocInfo, SearchHit } from '../api'
import DocumentViewer from './DocumentViewer'
import Highlight from './Highlight'

interface Props {
  documents: DocInfo[]
}

const EXAMPLES = ['mortgage', '切块大小', '第 60 页']

export default function SearchPanel({ documents }: Props) {
  const [query, setQuery] = useState('')
  const [docId, setDocId] = useState('')
  const [hits, setHits] = useState<SearchHit[] | null>(null)
  const [related, setRelated] = useState<SearchHit[]>([])
  const [terms, setTerms] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [lastQuery, setLastQuery] = useState('')
  const [viewer, setViewer] = useState<SearchHit | null>(null)

  const submit = async (raw?: string) => {
    const text = (raw ?? query).trim()
    if (!text || loading) return
    setLoading(true)
    setError('')
    try {
      const result = await searchChunks(text, docId || undefined, 12)
      setHits(result.hits)
      setRelated(result.related ?? [])
      setTerms(result.terms ?? [])
      setLastQuery(text)
      setQuery(text)
    } catch (err) {
      setError((err as Error).message)
      setHits(null)
      setRelated([])
    } finally {
      setLoading(false)
    }
  }

  const renderHit = (hit: SearchHit, semantic = false) => (
    <div
      key={hit.chunk_id}
      role="button"
      tabIndex={0}
      onClick={() => setViewer(hit)}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') setViewer(hit)
      }}
      className="cursor-pointer rounded-xl border border-slate-200 bg-white px-4 py-3 transition hover:border-purple-300 hover:shadow-sm"
      title="点击查看原文这一页"
    >
      <div className="flex items-center justify-between gap-2 text-xs">
        <span className="flex min-w-0 items-center gap-1.5 font-medium text-slate-700">
          <FileText size={13} className="shrink-0 text-purple-500" />
          <span className="truncate">{hit.source}</span>
          {hit.page ? <span className="shrink-0 text-slate-400">第 {hit.page} 页</span> : null}
        </span>
        <span className="shrink-0 text-[11px] text-slate-300">
          {semantic ? '语义相关' : '精确命中'} · {hit.score.toFixed(3)}
        </span>
      </div>
      <p className="mt-1.5 text-[13px] leading-relaxed text-slate-600">
        <Highlight text={hit.snippet} terms={semantic ? [] : hit.matched_terms} />
      </p>
      <p className="mt-1 text-[11px] text-slate-300">点击查看原文 →</p>
    </div>
  )

  return (
    <div className="flex h-full flex-col">
      <div className="mb-3">
        <h2 className="text-lg font-semibold text-slate-800">原文搜索</h2>
        <p className="mt-0.5 text-sm text-slate-500">
          只返回真正包含关键词的原文；点结果直接跳到对应页。
        </p>
      </div>

      <form
        onSubmit={(event) => {
          event.preventDefault()
          void submit()
        }}
        className="flex flex-col gap-2 sm:flex-row"
      >
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="搜索词、短语或页码，例如 mortgage、切块大小"
          className="h-10 flex-1 rounded-xl border border-slate-300 bg-white px-3 text-sm outline-none focus:border-purple-400 focus:ring-2 focus:ring-purple-100"
        />
        <select
          value={docId}
          onChange={(event) => setDocId(event.target.value)}
          className="h-10 rounded-xl border border-slate-300 bg-white px-2 text-sm text-slate-700 outline-none focus:border-purple-400"
        >
          <option value="">全部文档</option>
          {documents.map((doc) => (
            <option key={doc.doc_id} value={doc.doc_id}>
              {doc.source}
            </option>
          ))}
        </select>
        <button
          type="submit"
          disabled={!query.trim() || loading}
          className="flex h-10 items-center justify-center gap-1.5 rounded-xl bg-purple-600 px-4 text-sm font-medium text-white transition hover:bg-purple-700 disabled:opacity-40"
        >
          {loading ? <Loader2 size={15} className="animate-spin" /> : <Search size={15} />}
          搜索
        </button>
      </form>

      <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-slate-400">
        <span>试试：</span>
        {EXAMPLES.map((item) => (
          <button
            key={item}
            onClick={() => void submit(item)}
            className="rounded-full border border-slate-200 px-2 py-0.5 hover:border-purple-300 hover:text-purple-600"
          >
            {item}
          </button>
        ))}
        {docId && <span className="text-purple-600">已在指定文档内检索</span>}
      </div>

      <div className="mt-3 flex-1 overflow-y-auto">
        {error && (
          <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            搜索失败：{error}
          </div>
        )}

        {!error && hits === null && (
          <div className="flex h-full flex-col items-center justify-center gap-3 py-16 text-center text-slate-400">
            <Sparkles size={28} />
            <p className="text-sm">
              输入关键词，只匹配真正出现过的原文
              <br />
              点结果可直接查看对应页并高亮
            </p>
          </div>
        )}

        {!error && hits !== null && hits.length === 0 && (
          <div className="space-y-3">
            <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
              {docId ? '这篇文档里' : '知识库里'}没有出现「{lastQuery}」。
              可以换个词，或把范围切回全部文档。
            </div>

            {related.length > 0 && (
              <>
                <p className="text-xs text-slate-400">
                  下面是语义相近、但<strong className="text-slate-500">不包含</strong>关键词的片段，仅供参考：
                </p>
                <div className="space-y-2 pb-4">
                  {related.map((hit) => renderHit(hit, true))}
                </div>
              </>
            )}
          </div>
        )}

        {!error && hits !== null && hits.length > 0 && (
          <div className="space-y-2 pb-4">
            <p className="text-xs text-slate-400">
              「{lastQuery}」命中 {hits.length} 个片段
            </p>
            {hits.map((hit) => renderHit(hit))}
          </div>
        )}
      </div>

      {viewer && (
        <DocumentViewer
          docId={viewer.doc_id}
          source={viewer.source}
          page={viewer.page}
          terms={viewer.matched_terms.length > 0 ? viewer.matched_terms : terms}
          snippet={viewer.snippet}
          onClose={() => setViewer(null)}
        />
      )}
    </div>
  )
}
