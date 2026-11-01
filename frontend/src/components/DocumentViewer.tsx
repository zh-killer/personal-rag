import { useEffect, useRef, useState } from 'react'
import { ExternalLink, FileText, Loader2, X } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { documentFileUrl, fetchPageText } from '../api'
import Highlight from './Highlight'

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function clearHighlights(root: HTMLElement): void {
  root.querySelectorAll('mark[data-search-hl]').forEach((mark) => {
    const parent = mark.parentNode
    if (!parent) return
    parent.replaceChild(document.createTextNode(mark.textContent ?? ''), mark)
    parent.normalize()
  })
}

/** 在已经渲染好的 Markdown DOM 里把命中词包成 <mark> */
function highlightText(root: HTMLElement, terms: string[]): void {
  clearHighlights(root)
  const valid = terms.filter((term) => term.trim().length > 0 && term.length <= 40)
  if (valid.length === 0) return

  const pattern = new RegExp(`(${valid.map(escapeRegExp).join('|')})`, 'gi')
  const test = new RegExp(valid.map(escapeRegExp).join('|'), 'i')
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)
  const targets: Text[] = []
  while (walker.nextNode()) {
    const node = walker.currentNode as Text
    if (!node.nodeValue || !test.test(node.nodeValue)) continue
    if (node.parentElement?.closest('pre')) continue
    targets.push(node)
  }

  for (const node of targets) {
    const parts = (node.nodeValue ?? '').split(pattern)
    if (parts.length <= 1) continue
    const fragment = document.createDocumentFragment()
    for (const part of parts) {
      if (!part) continue
      if (valid.some((term) => part.toLowerCase() === term.toLowerCase())) {
        const mark = document.createElement('mark')
        mark.dataset.searchHl = '1'
        mark.className = 'rounded bg-yellow-200 px-0.5 text-slate-900'
        mark.textContent = part
        fragment.appendChild(mark)
      } else {
        fragment.appendChild(document.createTextNode(part))
      }
    }
    node.parentNode?.replaceChild(fragment, node)
  }
}

interface Props {
  docId: string
  source: string
  page: number | null
  terms: string[]
  snippet?: string
  onClose: () => void
}

/** 点开搜索结果：直接看原文第 N 页，并高亮关键词 */
export default function DocumentViewer({ docId, source, page, terms, snippet, onClose }: Props) {
  const [text, setText] = useState('')
  const [markdown, setMarkdown] = useState('')
  const [loading, setLoading] = useState(false)
  const [tab, setTab] = useState<'file' | 'text'>('file')
  const [imageError, setImageError] = useState(false)
  const markdownRef = useRef<HTMLDivElement>(null)

  const lower = source.toLowerCase()
  const isPdf = lower.endsWith('.pdf')
  const isImage = /\.(png|jpe?g|webp|bmp)$/.test(lower)
  const canShowFile = isPdf || isImage
  const fileUrl = documentFileUrl(docId, page)
  const termQuery =
    terms.length > 0 ? `?terms=${encodeURIComponent(terms.join(','))}` : ''
  const imageUrl = page
    ? `/api/documents/${encodeURIComponent(docId)}/page/${page}/image${termQuery}`
    : ''

  useEffect(() => {
    setImageError(false)
  }, [docId, page])

  useEffect(() => {
    if (markdownRef.current) highlightText(markdownRef.current, terms)
  }, [markdown, terms])

  useEffect(() => {
    if (!page) {
      setTab('text')
      return
    }
    setTab('file')
    let cancelled = false
    setLoading(true)
    fetchPageText(docId, page)
      .then((data) => {
        if (!cancelled) {
          setText(data.text)
          setMarkdown(data.markdown || '')
        }
      })
      .catch(() => {
        if (!cancelled) setText('')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [docId, page])

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 p-4"
      onClick={onClose}
    >
      <div
        className="flex h-[85vh] w-full max-w-4xl flex-col overflow-hidden rounded-2xl bg-white shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-center justify-between gap-3 border-b border-slate-200 px-4 py-2.5">
          <div className="flex min-w-0 items-center gap-2 text-sm">
            <FileText size={16} className="shrink-0 text-purple-500" />
            <span className="truncate font-medium text-slate-800">{source}</span>
            {page ? (
              <span className="shrink-0 rounded bg-slate-100 px-1.5 py-0.5 text-[11px] text-slate-500">
                第 {page} 页
              </span>
            ) : null}
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {isPdf && imageUrl && (
              <a
                href={imageUrl}
                target="_blank"
                rel="noreferrer"
                className="flex items-center gap-1 text-xs text-slate-500 hover:underline"
              >
                查看大图
              </a>
            )}
            <a
              href={fileUrl}
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1 text-xs text-purple-600 hover:underline"
            >
              <ExternalLink size={13} /> 新标签打开
            </a>
            <button
              onClick={onClose}
              className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
              aria-label="关闭"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {canShowFile && (
          <div className="flex gap-1 border-b border-slate-100 px-4 py-1.5 text-xs">
            <button
              onClick={() => setTab('file')}
              className={`rounded px-2 py-0.5 ${
                tab === 'file' ? 'bg-slate-800 text-white' : 'text-slate-500 hover:text-slate-800'
              }`}
            >
              原文
            </button>
            <button
              onClick={() => setTab('text')}
              className={`rounded px-2 py-0.5 ${
                tab === 'text' ? 'bg-slate-800 text-white' : 'text-slate-500 hover:text-slate-800'
              }`}
            >
              文字预览
            </button>
          </div>
        )}

        <div className="min-h-0 flex-1 bg-slate-50">
          {canShowFile && tab === 'file' ? (
            isPdf ? (
              <div className="h-full overflow-auto p-2">
                {imageError || !imageUrl ? (
                  <div className="flex h-full flex-col items-center justify-center gap-2 text-sm text-slate-400">
                    <p>页面渲染失败。</p>
                    <a
                      href={fileUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="text-purple-600 underline"
                    >
                      用新标签打开 PDF
                    </a>
                  </div>
                ) : (
                  <img
                    src={imageUrl}
                    alt={`${source} 第 ${page} 页`}
                    className="mx-auto max-w-full rounded-lg bg-white shadow"
                    onError={() => setImageError(true)}
                  />
                )}
              </div>
            ) : (
              <div className="flex h-full items-center justify-center overflow-auto p-4">
                <img src={fileUrl} alt={source} className="max-h-full max-w-full object-contain" />
              </div>
            )
          ) : (
            <div className="h-full overflow-y-auto px-5 py-6 sm:px-8">
              {loading ? (
                <div className="flex items-center gap-2 text-sm text-slate-400">
                  <Loader2 size={16} className="animate-spin" /> 正在读取第 {page} 页…
                </div>
              ) : (
                <div className="mx-auto max-w-2xl">
                  {snippet && (
                    <div className="mb-5 rounded-xl border border-amber-200 bg-amber-50 px-3.5 py-2.5 text-[12px] leading-relaxed text-amber-900">
                      <p className="mb-1 font-medium">命中片段</p>
                      <Highlight text={snippet} terms={terms} />
                    </div>
                  )}

                  {markdown ? (
                    <div className="md-body" ref={markdownRef}>
                      <ReactMarkdown>{markdown}</ReactMarkdown>
                    </div>
                  ) : text ? (
                    <div className="whitespace-pre-wrap text-[15px] leading-8 text-slate-700">
                      <Highlight text={text} terms={terms} />
                    </div>
                  ) : (
                    <p className="text-sm text-slate-400">这一页没有可显示的文本。</p>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
