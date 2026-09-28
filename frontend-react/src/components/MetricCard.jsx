import { FOM, fmtFom } from '../api'

function DeltaBadge({ metric, base }) {
  if (base == null || Number.isNaN(base)) return null
  const d = metric - base
  if (Math.abs(d) < 1e-9) return null
  const good = metric.better === 1 ? d < 0 : d > 0
  const cls = good
    ? 'text-[var(--green-hi)] bg-[rgba(52,211,153,0.12)] border-[rgba(52,211,153,0.35)]'
    : 'text-[var(--red)] bg-[rgba(251,113,133,0.1)] border-[rgba(251,113,133,0.3)]'
  const arrow = (metric.better === 1 ? d < 0 : d > 0) ? '▲' : '▼'
  const mag = metric.pct
    ? `${Math.abs(d * 100).toFixed(1)}pp`
    : Math.abs(d) >= 10 ? Math.abs(d).toFixed(0) : Math.abs(d).toFixed(2)
  return (
    <span className={`text-[9px] px-1 py-px rounded border ${cls}`}>
      {arrow} {mag} vs fixed
    </span>
  )
}

export function MetricCard({ fomKey, value, base }) {
  const m = FOM.find((f) => f.key === fomKey)
  if (!m) return null
  return (
    <div className="panel px-3 py-2 min-w-[130px]" title={m.def}>
      <div className="text-[9px] tracking-[0.18em] uppercase text-[var(--dim)]">{m.short}</div>
      <div className="text-lg text-[var(--green-hi)] glow-green tabular-nums leading-tight">
        {fmtFom(fomKey, value)}
      </div>
      <div className="mt-0.5"><DeltaBadge metric={m} base={base} /></div>
    </div>
  )
}

// The full PS figures-of-merit board: every metric, defined, with deltas vs the
// fixed-sweep baseline on the same seed.
export default function FoMPanel() {
  return null // replaced by inline grid in MissionControl (kept for API symmetry)
}
