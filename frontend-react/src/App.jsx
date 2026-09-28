import { useEffect, useState } from 'react'
import { getHealth, getReceiver } from './api'
import { AppProvider, useApp } from './store'
import RunControls from './components/RunControls'
import MissionControl from './pages/MissionControl'
import WaveformLab from './pages/WaveformLab'
import DecisionInspector from './pages/DecisionInspector'
import Benchmark from './pages/Benchmark'

const TABS = [
  { id: 'mission', label: 'Mission Control', el: <MissionControl /> },
  { id: 'waveform', label: 'Waveform Lab', el: <WaveformLab /> },
  { id: 'decision', label: 'Decision Inspector', el: <DecisionInspector /> },
  { id: 'benchmark', label: 'Benchmark', el: <Benchmark /> },
]

function HUD() {
  const { run, cursor, T, slotS, dwell, dwellIdx, nDwells, params, error } = useApp()
  return (
    <div className="flex items-center gap-4 text-[10px] tracking-[0.14em] uppercase flex-wrap">
      <span className="text-[var(--dim)]">
        scenario <span className="text-[var(--cyan)] normal-case tracking-normal">
          {params.split}/ep{params.seedIndex} · {params.nEmitters[0]}–{params.nEmitters[1]} emitters
        </span>
      </span>
      {run && (
        <>
          <span className="text-[var(--dim)]">
            mission <span className="text-[var(--green-hi)] normal-case tracking-normal tabular-nums">
              {(cursor * slotS).toFixed(1)}s / {(T * slotS).toFixed(0)}s
            </span>
          </span>
          <span className="text-[var(--dim)]">
            dwell <span className="text-[var(--green-hi)] normal-case tracking-normal tabular-nums">
              {dwellIdx + 1}/{nDwells}
            </span>
          </span>
          {dwell && (
            <span className="text-[var(--dim)]">
              scan <span className="text-[var(--cyan)] normal-case tracking-normal">band {dwell.band}</span>
              <span className={`ml-2 px-1.5 rounded ${dwell.hit
                ? 'bg-[var(--green)] text-[#052e21] font-bold'
                : 'bg-[rgba(251,113,133,0.15)] text-[var(--red)]'}`}>
                {dwell.hit ? 'HIT' : 'MISS'}
              </span>
            </span>
          )}
        </>
      )}
      {error && <span className="text-[var(--red)] normal-case">⚠ {error}</span>}
    </div>
  )
}

function Shell() {
  const [tab, setTab] = useState('mission')
  const [api, setApi] = useState(false)
  const { setBands } = useApp()

  useEffect(() => {
    getHealth().then(() => setApi(true)).catch(() => setApi(false))
    getReceiver().then(setBands).catch(() => {})
    const iv = setInterval(() => getHealth().then(() => setApi(true)).catch(() => setApi(false)), 5000)
    return () => clearInterval(iv)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="relative z-10 max-w-[1500px] mx-auto px-4 py-4 flex flex-col gap-3 min-h-full">
      <header className="flex items-end justify-between flex-wrap gap-2">
        <div>
          <h1 className="text-xl font-bold tracking-[0.08em] text-[var(--green-hi)] glow-green">
            SMARTSCAN <span className="text-[var(--dim)] font-normal">//</span>{' '}
            <span className="text-[var(--text)]">ES RECEIVER MISSION CONTROL</span>
          </h1>
          <div className="text-[10px] tracking-[0.2em] text-[var(--dim)] uppercase mt-0.5">
            SIH26055 · ML-based electronic-support scan scheduler · learns from hit/miss feedback only
          </div>
        </div>
        <div className="flex items-center gap-2 text-[10px] tracking-widest uppercase">
          <span className={`led ${api ? 'led-on' : 'led-off'}`} />
          <span className={api ? 'text-[var(--green-hi)]' : 'text-[var(--red)]'}>
            {api ? 'engine online' : 'engine offline — start uvicorn'}
          </span>
        </div>
      </header>

      <HUD />
      <RunControls />

      <nav className="flex gap-1.5">
        {TABS.map((t) => (
          <button key={t.id} className={`tab-btn ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>

      <main className="flex-1">{TABS.find((t) => t.id === tab)?.el}</main>

      <footer className="text-[9px] text-[var(--dim)] tracking-wider pb-1">
        Scheduler observes only hit/miss + amplitude of the band it scanned — ground truth shown in overlays is
        demo-only. Baselines: fixed sweep / random / greedy · Proposed: UCB & Thompson contextual bandits.
      </footer>
    </div>
  )
}

export default function App() {
  return (
    <AppProvider>
      <Shell />
    </AppProvider>
  )
}
