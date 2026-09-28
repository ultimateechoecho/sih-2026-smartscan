import { useMemo } from 'react'
import { hitRateCurve, useApp } from '../store'
import { ALGO_LABEL } from '../api'

// Live duel: running hit-rate of the chosen scheduler vs the fixed-sweep
// baseline on the SAME scenario seed — the impact story, updating during replay.
export default function HitRateDuel() {
  const { run, dwellIdx } = useApp()
  const mine = useMemo(() => hitRateCurve(run?.series), [run])
  const fixed = useMemo(() => hitRateCurve(run?.fixed_series), [run])

  if (!mine || !fixed) return null
  const i = Math.max(0, Math.min(dwellIdx, mine.length - 1))
  const j = Math.max(0, Math.min(dwellIdx, fixed.length - 1))
  const my = mine[i] ?? 0
  const fy = fixed[j] ?? 0
  const ratio = run?.metrics?.interception_ratio
  const fr = run?.fixed_metrics?.interception_ratio

  return (
    <div className="space-y-2">
      <div>
        <div className="flex justify-between text-[11px] mb-1">
          <span className="text-[var(--green-hi)]">{ALGO_LABEL[run.algo] || run.algo} — running hit rate</span>
          <span className="tabular-nums text-[var(--green-hi)] glow-green">{(my * 100).toFixed(1)}%</span>
        </div>
        <div className="h-2.5 rounded bg-black/50 border border-[rgba(52,211,153,0.2)] overflow-hidden">
          <div className="h-full transition-[width] duration-100"
            style={{ width: `${my * 100}%`, background: 'linear-gradient(90deg,#0d9488,#6ee7b7)' }} />
        </div>
      </div>
      <div>
        <div className="flex justify-between text-[11px] mb-1">
          <span className="text-[var(--dim)]">Fixed sweep (same seed)</span>
          <span className="tabular-nums text-[var(--dim)]">{(fy * 100).toFixed(1)}%</span>
        </div>
        <div className="h-2.5 rounded bg-black/50 border border-[rgba(148,163,184,0.15)] overflow-hidden">
          <div className="h-full transition-[width] duration-100"
            style={{ width: `${fy * 100}%`, background: 'linear-gradient(90deg,#475569,#94a3b8)' }} />
        </div>
      </div>
      {ratio != null && fr != null && (
        <div className="text-[11px] pt-1 text-[var(--text)]">
          Burst interception ratio:{' '}
          <span className="text-[var(--green-hi)] font-bold glow-green">{(ratio * 100).toFixed(1)}%</span>
          <span className="text-[var(--dim)]"> vs fixed {(fr * 100).toFixed(1)}% </span>
          <span className={ratio >= fr ? 'text-[var(--green-hi)]' : 'text-[var(--red)]'}>
            ({ratio >= fr ? '+' : ''}{((ratio - fr) * 100).toFixed(1)}pp)
          </span>
        </div>
      )}
    </div>
  )
}
