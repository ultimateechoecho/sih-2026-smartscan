import { FOM, fmtFom } from '../api'

// Full figures-of-merit board — every PS metric with a one-line definition
// (tooltip) and a delta vs the fixed-sweep baseline on the same seed.
export default function FoMGrid({ metrics, fixedMetrics }) {
  return (
    <div className="grid gap-2" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))' }}>
      {FOM.map((m) => {
        const v = metrics?.[m.key]
        const b = fixedMetrics?.[m.key]
        let delta = null
        if (v != null && b != null && !Number.isNaN(v) && !Number.isNaN(b) && m.better != null) {
          const d = v - b
          const good = m.better === 1 ? d < 0 : d > 0
          delta = (
            <span className={`text-[9px] ${good ? 'text-[var(--green-hi)]' : 'text-[var(--red)]'}`}>
              {good ? '▲' : '▼'} {m.pct ? `${Math.abs(d * 100).toFixed(1)}pp`
                : Math.abs(d) >= 10 ? Math.abs(d).toFixed(0) : Math.abs(d).toFixed(2)}
              <span className="text-[var(--dim)]"> vs fixed</span>
            </span>
          )
        }
        return (
          <div key={m.key} className="panel px-2.5 py-1.5" title={m.def}>
            <div className="text-[9px] tracking-[0.16em] uppercase text-[var(--dim)] truncate">{m.short}</div>
            <div className="text-base text-[var(--green-hi)] glow-green tabular-nums leading-tight">
              {fmtFom(m.key, v)}
            </div>
            <div className="h-3.5">{delta}</div>
          </div>
        )
      })}
    </div>
  )
}
