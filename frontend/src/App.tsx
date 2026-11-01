import { useEffect, useState } from 'react'
import { Brain, FolderOpen, MessageCircle, Search, Upload, WifiOff } from 'lucide-react'
import ChatPanel from './components/ChatPanel'
import DocListPanel from './components/DocListPanel'
import SearchPanel from './components/SearchPanel'
import UploadPanel from './components/UploadPanel'
import { fetchDocuments, fetchHealth } from './api'
import type { DocInfo, Health } from './api'

type Tab = 'chat' | 'search' | 'upload' | 'docs'

const TABS: { key: Tab; label: string; icon: typeof MessageCircle }[] = [
  { key: 'chat', label: '问答', icon: MessageCircle },
  { key: 'search', label: '搜索', icon: Search },
  { key: 'upload', label: '上传', icon: Upload },
  { key: 'docs', label: '文档', icon: FolderOpen },
]

export default function App() {
  const [tab, setTab] = useState<Tab>('chat')
  const [health, setHealth] = useState<Health | null>(null)
  const [documents, setDocuments] = useState<DocInfo[]>([])
  const [offline, setOffline] = useState(false)
  const [refresh, setRefresh] = useState(0)

  useEffect(() => {
    let cancelled = false
    fetchHealth()
      .then((data) => {
        if (cancelled) return
        setHealth(data)
        setOffline(false)
      })
      .catch(() => {
        if (cancelled) return
        setHealth(null)
        setOffline(true)
      })
    fetchDocuments()
      .then((docs) => {
        if (!cancelled) setDocuments(docs)
      })
      .catch(() => {
        if (!cancelled) setDocuments([])
      })
    return () => {
      cancelled = true
    }
  }, [refresh])

  return (
    <div className="flex h-screen flex-col bg-slate-100 text-slate-900">
      <header className="shrink-0 border-b border-slate-200 bg-white">
        <div className="mx-auto flex h-14 w-full max-w-5xl items-center justify-between px-4">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-purple-600 text-white">
              <Brain size={18} />
            </div>
            <div className="leading-tight">
              <p className="text-sm font-semibold text-slate-800">PersonalRAG</p>
              <p className="text-[11px] text-slate-400">个人知识库问答</p>
            </div>
          </div>

          <div className="flex items-center gap-3 text-[11px] text-slate-500">
            {health && (
              <>
                <span className="hidden sm:inline">{health.model}</span>
                <span className="hidden sm:inline text-slate-300">|</span>
                <span>
                  {health.documents} 文档 · {health.chunks} 片段
                </span>
                <span
                  className={`flex items-center gap-1.5 rounded-full px-2 py-1 ${
                    health.llm_configured
                      ? 'bg-emerald-50 text-emerald-600'
                      : 'bg-amber-50 text-amber-600'
                  }`}
                >
                  <span
                    className={`h-1.5 w-1.5 rounded-full ${
                      health.llm_configured ? 'bg-emerald-500' : 'bg-amber-500'
                    }`}
                  />
                  {health.llm_configured ? '就绪' : '缺 API Key'}
                </span>
              </>
            )}
            {!health && !offline && <span className="text-slate-400">连接中…</span>}
          </div>
        </div>
      </header>

      <nav className="shrink-0 border-b border-slate-200 bg-white">
        <div className="mx-auto flex w-full max-w-5xl gap-1 px-4">
          {TABS.map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              onClick={() => setTab(key)}
              className={`-mb-px flex items-center gap-1.5 border-b-2 px-4 py-2.5 text-sm transition ${
                tab === key
                  ? 'border-purple-600 font-medium text-purple-700'
                  : 'border-transparent text-slate-500 hover:text-slate-800'
              }`}
            >
              <Icon size={16} />
              {label}
              {key === 'docs' && health ? (
                <span className="ml-0.5 rounded bg-slate-100 px-1.5 text-[11px] text-slate-500">
                  {health.documents}
                </span>
              ) : null}
            </button>
          ))}
        </div>
      </nav>

      {offline && (
        <div className="shrink-0 border-b border-red-200 bg-red-50 px-4 py-2 text-xs text-red-700">
          <div className="mx-auto flex w-full max-w-5xl items-center gap-2">
            <WifiOff size={13} />
            连不上后端：请先运行 <code className="rounded bg-red-100 px-1">python run.py</code>，
            再刷新页面。接口地址 <code className="rounded bg-red-100 px-1">/api/health</code>。
          </div>
        </div>
      )}

      <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col overflow-hidden px-3 py-4 sm:px-4">
        {/* 三个面板都保持挂载：切 Tab 不会丢掉正在进行的对话 / 上传列表 */}
        <section
          className={tab === 'chat' ? 'flex min-h-0 flex-1 flex-col' : 'hidden'}
          aria-hidden={tab !== 'chat'}
        >
          <ChatPanel
            llmConfigured={health?.llm_configured ?? true}
            documentCount={health?.documents ?? 0}
            documents={documents}
            onGoUpload={() => setTab('upload')}
          />
        </section>

        <section
          className={tab === 'search' ? 'flex h-full min-h-0 flex-col' : 'hidden'}
          aria-hidden={tab !== 'search'}
        >
          <SearchPanel documents={documents} />
        </section>

        <section
          className={tab === 'upload' ? 'h-full overflow-y-auto pb-4' : 'hidden'}
          aria-hidden={tab !== 'upload'}
        >
          <UploadPanel
            onUploaded={() => setRefresh((value) => value + 1)}
            onGoChat={() => setTab('chat')}
          />
        </section>

        <section
          className={tab === 'docs' ? 'h-full overflow-y-auto pb-4' : 'hidden'}
          aria-hidden={tab !== 'docs'}
        >
          <DocListPanel refreshTrigger={refresh} onGoUpload={() => setTab('upload')} />
        </section>
      </main>

      <footer className="shrink-0 border-t border-slate-200 bg-white py-2 text-center text-[11px] text-slate-400">
        PersonalRAG 2.0 · FastAPI + ChromaDB + BM25 · 回答严格基于已索引文档
      </footer>
    </div>
  )
}
