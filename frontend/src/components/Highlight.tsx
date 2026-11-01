interface Props {
  text: string
  terms: string[]
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/** 在文本里高亮命中词（大小写不敏感） */
export default function Highlight({ text, terms }: Props) {
  const valid = terms.filter((term) => term.length > 0)
  if (valid.length === 0) return <>{text}</>

  const pattern = new RegExp(`(${valid.map(escapeRegExp).join('|')})`, 'gi')
  const lowered = new Set(valid.map((term) => term.toLowerCase()))
  return (
    <>
      {text.split(pattern).map((part, index) =>
        lowered.has(part.toLowerCase()) ? (
          <mark key={index} className="rounded bg-yellow-100 px-0.5 text-slate-900">
            {part}
          </mark>
        ) : (
          <span key={index}>{part}</span>
        ),
      )}
    </>
  )
}
