import { useCallback, useRef, useState } from 'react'
import { AlertCircle, CheckCircle2, FileText, Loader2, UploadCloud, X } from 'lucide-react'
import { uploadFile, waitForTask } from '../api'

type Status = 'pending' | 'uploading' | 'done' | 'error'

interface Item {
  id: string
  file: File
  status: Status
  progress: number
  stage: string
  message?: string
  chunks?: number
}

interface Props {
  onUploaded: () => void
  onGoChat: () => void
}

const ACCEPT = '.pdf,.md,.markdown,.txt,.png,.jpg,.jpeg,.webp,.bmp'
const MAX_MB = 50

function newId(): string {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2)
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

export default function UploadPanel({ onUploaded, onGoChat }: Props) {
  const [items, setItems] = useState<Item[]>([])
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const patch = (id: string, changes: Partial<Item>) => {
    setItems((prev) => prev.map((item) => (item.id === id ? { ...item, ...changes } : item)))
  }

  const addFiles = useCallback((files: File[]) => {
    const next: Item[] = files.map((file) => {
      const suffix = `.${file.name.split('.').pop()?.toLowerCase() ?? ''}`
      if (!ACCEPT.split(',').includes(suffix)) {
        return {
          id: newId(),
          file,
          status: 'error' as Status,
          progress: 0,
          stage: '',
          message: `不支持的格式（${suffix}）`,
        }
      }
      if (file.size > MAX_MB * 1024 * 1024) {
        return {
          id: newId(),
          file,
          status: 'error' as Status,
          progress: 0,
          stage: '',
          message: `超过 ${MAX_MB} MB 限制`,
        }
      }
      return { id: newId(), file, status: 'pending' as Status, progress: 0, stage: '' }
    })
    setItems((prev) => [...prev, ...next])
  }, [])

  const onDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragging(false)
    addFiles(Array.from(event.dataTransfer.files))
  }

  const start = async () => {
    const pending = items.filter((item) => item.status === 'pending')
    if (!pending.length || uploading) return
    setUploading(true)
    let succeeded = 0

    for (const item of pending) {
      patch(item.id, { status: 'uploading', progress: 0, stage: '上传中', message: undefined })
      try {
        const accepted = await uploadFile(item.file)
        const task = await waitForTask(accepted.task_id, (progress) => {
          patch(item.id, {
            progress: progress.progress,
            stage: progress.stage || '处理中',
          })
        })
        if (task.status === 'error') {
          throw new Error(task.error || '入库失败')
        }
        patch(item.id, {
          status: 'done',
          progress: 1,
          stage: task.stage,
          chunks: task.chunk_count,
          message: task.message,
        })
        succeeded += 1
      } catch (error) {
        patch(item.id, {
          status: 'error',
          stage: '',
          message: (error as Error).message,
        })
      }
    }

    setUploading(false)
    if (succeeded > 0) onUploaded()
  }

  const pendingCount = items.filter((item) => item.status === 'pending').length

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-lg font-semibold text-slate-800">上传知识文档</h2>
        <p className="mt-1 text-sm text-slate-500">
          支持 PDF / Markdown / TXT / 图片；扫描件与图片会自动 OCR（较慢，看进度条）。
        </p>
      </div>

      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(event) => {
          event.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`cursor-pointer rounded-2xl border-2 border-dashed px-6 py-12 text-center transition ${
          dragging
            ? 'border-purple-400 bg-purple-50'
            : 'border-slate-300 bg-white hover:border-purple-300 hover:bg-purple-50/40'
        }`}
      >
        <UploadCloud size={40} className="mx-auto mb-3 text-slate-400" />
        <p className="text-sm font-medium text-slate-700">拖拽文件到这里，或点击选择</p>
        <p className="mt-1 text-xs text-slate-400">可一次选择多个文件，单文件最大 {MAX_MB} MB</p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPT}
          className="hidden"
          onChange={(event) => {
            addFiles(Array.from(event.target.files ?? []))
            event.target.value = ''
          }}
        />
      </div>

      {items.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-slate-700">文件列表（{items.length}）</h3>
            {!uploading && items.some((item) => item.status === 'done') && (
              <button
                onClick={() => setItems((prev) => prev.filter((item) => item.status !== 'done'))}
                className="text-xs text-slate-400 hover:text-slate-600"
              >
                清空已完成
              </button>
            )}
          </div>

          {items.map((item) => (
            <div
              key={item.id}
              className="rounded-xl border border-slate-200 bg-white px-3 py-2.5"
            >
              <div className="flex items-center justify-between gap-3">
                <div className="flex min-w-0 items-center gap-2.5">
                  <FileText size={18} className="shrink-0 text-slate-400" />
                  <div className="min-w-0">
                    <p className="truncate text-sm text-slate-700">{item.file.name}</p>
                    <p className="text-xs text-slate-400">
                      {formatSize(item.file.size)}
                      {item.status === 'uploading' && (
                        <span className="ml-2 text-purple-600">
                          {item.stage} {Math.round(item.progress * 100)}%
                        </span>
                      )}
                      {item.status === 'done' && (
                        <span className="ml-2 text-emerald-600">
                          已索引 {item.chunks} 个片段
                        </span>
                      )}
                      {item.status === 'error' && (
                        <span className="ml-2 text-red-500">{item.message}</span>
                      )}
                    </p>
                  </div>
                </div>

                <div className="flex shrink-0 items-center gap-2">
                  {item.status === 'uploading' && (
                    <Loader2 size={16} className="animate-spin text-purple-500" />
                  )}
                  {item.status === 'done' && <CheckCircle2 size={16} className="text-emerald-500" />}
                  {item.status === 'error' && <AlertCircle size={16} className="text-red-500" />}
                  {item.status === 'pending' && (
                    <button
                      onClick={() => setItems((prev) => prev.filter((it) => it.id !== item.id))}
                      className="text-slate-300 hover:text-red-500"
                      aria-label="移除"
                    >
                      <X size={16} />
                    </button>
                  )}
                </div>
              </div>

              {item.status === 'uploading' && (
                <div className="mt-2 h-1 w-full overflow-hidden rounded-full bg-slate-100">
                  <div
                    className="h-full rounded-full bg-purple-500 transition-all"
                    style={{ width: `${Math.max(3, Math.round(item.progress * 100))}%` }}
                  />
                </div>
              )}
            </div>
          ))}

          <button
            onClick={() => void start()}
            disabled={pendingCount === 0 || uploading}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-purple-600 py-2.5 text-sm font-medium text-white transition hover:bg-purple-700 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {uploading ? (
              <>
                <Loader2 size={16} className="animate-spin" /> 正在入库…
              </>
            ) : (
              `开始索引${pendingCount ? `（${pendingCount}）` : ''}`
            )}
          </button>

          {items.some((item) => item.status === 'done') && (
            <button
              onClick={onGoChat}
              className="w-full rounded-xl border border-purple-200 bg-purple-50 py-2.5 text-sm font-medium text-purple-700 transition hover:bg-purple-100"
            >
              已加入知识库，去提问 →
            </button>
          )}
        </div>
      )}

      <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-xs leading-relaxed text-slate-500">
        <p className="font-medium text-slate-600">索引说明</p>
        <p className="mt-1">
          上传后立即返回，解析/OCR/向量化在后台进行，关掉页面也不会中断。
          同名文件重传会更新，内容完全相同的文件会跳过。扫描件会按页 OCR，页数多时会慢一些。
        </p>
      </div>
    </div>
  )
}
