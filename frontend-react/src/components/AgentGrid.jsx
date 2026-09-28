import { useState } from 'react'
import { useApp } from '../store'

// 6×6 band grid coloured by the metric of your choice (due / periodic conf /
// priority / score).  Click a band to lock the Waveform Lab oscilloscope on it.
const METRICS = [
  { key: 'score', label: 'decision score' },
  { key: 'due', label: 'due (agents)' },
  { key: 'periodic_conf', label: 'periodicity conf' },
  { key: 'priority', label: 'priority' },
  { key: 'hit_rate', label: 'hit rate' },
  { key: 'visit_age', label: 'staleness' },
]

function color(v) {
  if (v == null || v <= 0.001) return 'rgba(52,211,153,0.06)'
  return `rgba(52,211,153,${0.15 + 0.8 * Math.min(1, v)})`
}

export default function AgentGrid() {
  const { decision, selectedBand, setSelectedBand, dwell } = useApp()
  const [metricKey, setMetricKey] = useState('score')
  const shown = selectedBand ?? dwell?.band ?? null
  const values = decision?.features?.[metricKey]
  const label = METRICS.find((m) => m.key === metricKey)?.label

  return (
    <div>
      <div className="grid grid-cols-6 gap-1 max-w-[240px]">
        {Array.from({ length: 36 }, (_, b) => {
          const v = values ? values[b] : 0
          const isShown = shown === b
          const isAction = decision?.action === b
          return (
            <button
              key={b}
              onClick={() => setSelectedBand(selectedBand === b ? null : b)}
              title={`B${b} · ${label}: ${typeof v === 'number' ? v.toFixed(2) : '—'}`}
              className="relative h-8 rounded border text-[8px] transition-all"
              style={{
                background: color(v),
                borderColor: isAction ? '#6ee7b7' : isShown ? '#38bdf8' : 'rgba(52,211,153,0.15)',
                boxShadow: isAction ? '0 0 8px rgba(110,231,183,0.6)' : 'none',
              }}
            >
              <span className={v > 0.55 ? 'text-black/80' : 'text-[var(--dim)]'}>{b}</span>
            </button>
          )
        })}
      </div>
      <div className="mt-2 flex items-center gap-2 text-[10px]">
        <select value={metricKey} onChange={(e) => setMetricKey(e.target.value)} className="field">
          {METRICS.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
        </select>
        <span className="text-[var(--dim)]">click a band to lock the scope</span>
      </div>
    </div>
  )
}
