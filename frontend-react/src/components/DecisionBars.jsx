import { useMemo } from 'react'
import { useApp } from '../store'
import { ALGO_LABEL, BASELINES } from '../api'

// Per-band decision score breakdown for the current dwell (bandit algos):
// stacked components Q / exploration / due / priority / staleness.
const COMP = [
  { key: 'Q', color: '#34d399', label: 'Q (value)' },
  { key: 'explore', color: '#38bdf8', label: 'explore (UCB)' },
  { key: 'due', color: '#fbbf24', label: 'due (agents)' },
  { key: 'priority', color: '#f472b6', label: 'priority (surprise)' },
  { key: 'stale', color: '#94a3b8', label: 'staleness' },
]

export default function DecisionBars() {
  const { decision, run } = useApp()
  const rows = useMemo(() => {
    if (!decision?.components) return null
    const c = decision.components
    return c.score
      .map((s, b) => ({ b, s, parts: COMP.map((k) => ({ ...k, v: c[k.key][b] })) }))
      .sort((x, y) => y.s - x.s)
      .slice(0, 8)
  }, [decision])

  if (!run) return null
  if (!rows) {
    return (
      <div className="text-[11px] text-[var(--dim)] leading-relaxed">
        “{ALGO_LABEL[run.algo]}” is an open-loop baseline — it has no per-band scores to inspect.
        Run <b className="text-[var(--green-hi)]">UCB</b> or <b className="text-[var(--green-hi)]">Thompson</b> to
        see the decision math live. Current reason:{' '}
        <span className="text-[var(--text)]">{decision?.reason ?? '—'}</span>
      </div>
    )
  }

  const max = Math.max(...rows.map((r) => r.s), 0.01)
  return (
    <div className="space-y-1.5">
      <div className="flex gap-3 flex-wrap text-[9px] text-[var(--dim)]">
        {COMP.map((k) => (
          <span key={k.key} className="flex items-center gap-1">
            <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: k.color }} />
            {k.label}
          </span>
        ))}
      </div>
      {rows.map((r) => {
        const chosen = r.b === decision.action
        return (
          <div key={r.b} className="flex items-center gap-2">
            <span
              className={`w-12 text-[10px] tabular-nums ${chosen ? 'text-[var(--green-hi)] font-bold' : 'text-[var(--dim)]'}`}
            >
              {chosen ? '▶ ' : '  '}B{String(r.b).padStart(2, '0')}
            </span>
            <div className="flex-1 h-4 bg-black/40 rounded border border-[rgba(52,211,153,0.12)] overflow-hidden flex">
              {r.parts.map((p, i) => (
                p.v > 0.002 && (
                  <div key={i} style={{ width: `${(p.v / max) * 100}%`, background: p.color, opacity: 0.85 }} />
                )
              ))}
            </div>
            <span className={`w-12 text-right text-[10px] tabular-nums ${chosen ? 'text-[var(--green-hi)] font-bold' : 'text-[var(--dim)]'}`}>
              {r.s.toFixed(2)}
            </span>
          </div>
        )
      })}
    </div>
  )
}
