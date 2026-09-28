import { useEffect, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { ALGOS, ALGO_LABEL, BASELINES, FOM, PRIMARY_FOM, fetchBenchmark, fmtFom } from '../api'
import Panel from '../components/Panel'

const AXIS = { stroke: '#5f7a70', fontSize: 10, fontFamily: 'ui-monospace, monospace' }
const TOOLTIP = {
  contentStyle: {
    background: '#0a1210', border: '1px solid rgba(52,211,153,0.3)', borderRadius: 8,
    fontSize: 11, fontFamily: 'ui-monospace, monospace', color: '#c9e5da',
  },
}

function barColor(a) {
  return BASELINES.has(a) ? '#64748b' : '#34d399'
}

// Tab 4 — paired benchmark across all five schedulers with the full PS
// figures-of-merit table.
export default function Benchmark() {
  const [data, setData] = useState(null)
  const [split, setSplit] = useState('test')
  const [episodes, setEpisodes] = useState(10)
  const [busy, setBusy] = useState(false)

  const load = async (s = split, eps = episodes, cache = true) => {
    setBusy(true)
    try {
      setData(await fetchBenchmark(s, eps, cache))
    } finally {
      setBusy(false)
    }
  }
  useEffect(() => { load() }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const rows = data
    ? ALGOS.map((a) => ({ algo: a, ...(data.mean[a] ?? {}) })).filter((r) => data.mean[r.algo])
    : []

  return (
    <div className="space-y-3">
      <Panel
        title="Paired-seed benchmark — all schedulers, held-out scenarios"
        right={
          <div className="flex items-center gap-2 text-[10px]">
            {data && (
              <span className={data.cached ? 'text-[var(--amber)]' : 'text-[var(--green-hi)]'}>
                {data.cached ? `cached · ${data.episodes} episodes (results/)` : `fresh run · ${data.episodes} episodes`}
              </span>
            )}
            <select className="field" value={split} onChange={(e) => setSplit(e.target.value)}>
              {['test', 'val', 'train'].map((s) => <option key={s}>{s}</option>)}
            </select>
            <input type="number" min={3} max={20} className="field w-14"
              value={episodes} onChange={(e) => setEpisodes(+e.target.value)} />
            <button className="btn-ghost" disabled={busy}
              onClick={() => load(split, episodes, false)}>
              {busy ? 'RUNNING…' : 'RUN FRESH'}
            </button>
            <button className="btn-ghost" disabled={busy} onClick={() => load(split, episodes, true)}>
              LOAD CACHED
            </button>
          </div>
        }
      >
        {!rows.length && <div className="text-[var(--dim)] py-6 text-[12px]">loading benchmark…</div>}
        {rows.length > 0 && (
          <>
            <div className="grid grid-cols-1 lg:grid-cols-2 2xl:grid-cols-4 gap-3">
              {PRIMARY_FOM.map((key) => {
                const m = FOM.find((f) => f.key === key)
                const chartData = rows.map((r) => ({ name: ALGO_LABEL[r.algo] ?? r.algo, v: r[key] }))
                return (
                  <div key={key} className="panel p-2" title={m.def}>
                    <div className="text-[9px] tracking-[0.18em] uppercase text-[var(--dim)] mb-1">{m.short}</div>
                    <ResponsiveContainer width="100%" height={150}>
                      <BarChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: -18 }}>
                        <CartesianGrid stroke="rgba(52,211,153,0.08)" vertical={false} />
                        <XAxis dataKey="name" tick={{ ...AXIS, fontSize: 8 }} interval={0} angle={-14}
                          dy={6} height={34} />
                        <YAxis tick={AXIS} />
                        <Tooltip {...TOOLTIP} formatter={(v) => fmtFom(key, v)} />
                        <Bar dataKey="v" radius={[3, 3, 0, 0]}>
                          {chartData.map((d, i) => (
                            <Cell key={i} fill={barColor(ALGOS[i])} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )
              })}
            </div>

            <div className="overflow-auto mt-3">
              <table className="roc w-full text-[11px] tabular-nums">
                <thead>
                  <tr>
                    <th>Metric</th>
                    {rows.map((r) => (
                      <th key={r.algo} className={BASELINES.has(r.algo) ? '' : 'text-[var(--green-hi)]'}>
                        {ALGO_LABEL[r.algo] ?? r.algo}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {FOM.map((m) => {
                    const vals = rows.map((r) => r[m.key]).filter((v) => v != null && !Number.isNaN(v))
                    const best = m.better == null || !vals.length ? null
                      : m.better === 1 ? Math.min(...vals) : Math.max(...vals)
                    return (
                      <tr key={m.key}>
                        <td className="text-[var(--dim)]" title={m.def}>{m.label}</td>
                        {rows.map((r) => {
                          const v = r[m.key]
                          const isBest = best != null && v === best
                          return (
                            <td key={r.algo} className={isBest ? 'text-[var(--green-hi)] font-bold glow-green' : ''}>
                              {fmtFom(m.key, v)}
                            </td>
                          )
                        })}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <div className="text-[9px] text-[var(--dim)] mt-2">
              Grey = open-loop baselines · green = proposed bandit schedulers · bold green = best per metric.
              All schedulers ran on identical scenario seeds (paired comparison).
            </div>
          </>
        )}
      </Panel>
    </div>
  )
}
